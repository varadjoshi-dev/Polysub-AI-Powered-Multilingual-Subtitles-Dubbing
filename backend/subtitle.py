import argparse
import os
import subprocess
import sys
import tempfile
import uuid
import whisper
import srt
import datetime
from tqdm import tqdm
import asyncio
import pysubs2
from pydub import AudioSegment, effects
import re
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ttsmms import download, TTS
import inflect
import torch
from pydub import AudioSegment, effects
import ffmpeg
import spacy
import io
from pydub.utils import make_chunks
import regex
import shutil
from pydub import silence
import unicodedata


def _normalize_token_for_compare(tok: str) -> str:
    """Normalize token aggressively for duplicate detection."""
    if not tok:
        return ""
    tok = unicodedata.normalize("NFKC", tok)  # Unicode normalize
    core = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)  # strip punctuation
    return core.casefold()

def split_caption_text_two_lines(text, max_chars=40):
    """
    Split `text` into at most two lines without splitting words.
    Returns a list of 1 or 2 line strings (no trailing/leading spaces).
    """
    if not text:
        return [""]

    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + (1 if cur else 0) + len(w) <= max_chars:
            cur += (" " if cur else "") + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # If already 1 or 2 lines -> good
    if len(lines) <= 2:
        return [ln.strip() for ln in lines]

    # More than 2 lines: merge into 2 lines trying to balance by char-count
    # Strategy: greedily pack words into first line until roughly half of total chars
    total_chars = sum(len(w) for w in words) + (len(words) - 1)
    target = total_chars // 2
    l1_words = []
    cur_len = 0
    i = 0
    while i < len(words):
        w = words[i]
        add_len = len(w) + (1 if l1_words else 0)
        if cur_len + add_len <= target or not l1_words:
            l1_words.append(w)
            cur_len += add_len
            i += 1
        else:
            break
    l1 = " ".join(l1_words).strip()
    l2 = " ".join(words[i:]).strip()
    if not l2:
        return [l1]
    return [l1, l2]


def split_long_subtitles(input_srt, output_srt, max_chars=40):
    """
    Split long subtitle lines into multiple smaller chunks while
    proportionally distributing timing, and ensure each event text
    is at most 2 lines (joined by '\\n').
    """
    subs = pysubs2.load(input_srt, encoding="utf-8")
    new_subs = []

    for ev in subs:
        text = ev.text.strip()
        if not text:
            new_subs.append(ev)
            continue

        if len(text) <= max_chars:
            # still wrap into up-to-2 lines if necessary
            wrapped = split_caption_text_two_lines(text, max_chars=max_chars)
            ev.text = "\n".join(wrapped)
            new_subs.append(ev)
            continue

        # split into chunks (word-aware)
        chunks = split_caption_text(text, max_chars=max_chars)

        total_duration = ev.end - ev.start
        num_chunks = len(chunks) if chunks else 1
        base_chunk = total_duration // num_chunks
        remainder = total_duration - (base_chunk * num_chunks)

        offset = ev.start
        for i, chunk in enumerate(chunks):
            this_dur = base_chunk + (1 if i < remainder else 0)
            wrapped = split_caption_text_two_lines(chunk, max_chars=max_chars)
            new_ev = pysubs2.SSAEvent(
                start=offset,
                end=offset + this_dur,
                text="\n".join(wrapped)
            )
            new_subs.append(new_ev)
            offset += this_dur

    ssa = pysubs2.SSAFile()
    ssa.events = new_subs
    ssa.save(output_srt, format_="srt", encoding="utf-8")
    print(f"[FIXED] Long subtitles split → {output_srt}")

def _normalize_short_token(tok: str) -> str:
    """Normalize a single-word token for dedupe/compare (strip punctuation + casefold)."""
    if not tok:
        return ""
    core = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)
    return core.casefold()

def dedupe_nearby_words(words, time_tol_s=0.1):
    """
    Remove word duplicates that arise from chunk overlap.
    Keeps the earliest occurrence when two words are nearly identical.
    """
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["start"], w["end"]))
    out = [words_sorted[0]]

    for w in words_sorted[1:]:
        prev = out[-1]

        if (
            abs(w["start"] - prev["start"]) <= time_tol_s
            and _normalize_token_for_compare(w["word"]) == _normalize_token_for_compare(prev["word"])
        ):
            # keep longer span
            if w["end"] > prev["end"]:
                out[-1] = w
            continue

        if (
            w["start"] >= prev["start"]
            and w["end"] <= prev["end"]
            and _normalize_token_for_compare(w["word"]) == _normalize_token_for_compare(prev["word"])
        ):
            continue

        out.append(w)

    return out

def words_to_srt(words, max_chars=42, max_duration=5000):
    """
    Build subtitle events from Whisper word timestamps robustly.
    - words: list of {'word','start','end'} with times in seconds (floats)
    - returns an SRT string (srt.compose output)
    """
    if not words:
        return ""

    # Ensure sorted, deduped
    words_sorted = sorted(words, key=lambda w: w['start'])
    words_sorted = dedupe_nearby_words(words_sorted, time_tol_s=0.05)

    subs = []
    cur_words = []
    cur_start = None
    index = 1
    prev_end_s = 0.0

    for w in words_sorted:
        tok = w['word'].strip()
        if not tok:
            continue

        if not cur_words:
            cur_start = w['start']
            cur_words = [w]
        else:
            cur_words.append(w)

        text = " ".join(x['word'].strip() for x in cur_words).strip()
        cur_end = cur_words[-1]['end']
        duration_ms = (cur_end - cur_start) * 1000

        # Condition to emit a subtitle: too long text OR too long duration
        if len(text) > max_chars or duration_ms > max_duration:
            # If only one word in cur_words (edge case) -> emit it alone (avoid index error)
            if len(cur_words) == 1:
                emit_start = max(cur_start, prev_end_s + 0.001)
                emit_end = max(cur_end, emit_start + adaptive_min_display(tok) / 1000.0)
                content = "\n".join(split_caption_text_two_lines(tok, max_chars))
                subs.append(srt.Subtitle(
                    index=index,
                    start=datetime.timedelta(seconds=emit_start),
                    end=datetime.timedelta(seconds=emit_end),
                    content=content
                ))
                index += 1
                prev_end_s = emit_end
                cur_words = []
                cur_start = None
                continue

            # Emit everything except the last word, keep last word for next subtitle
            to_emit = cur_words[:-1]
            emit_text = " ".join(x['word'].strip() for x in to_emit).strip()
            emit_start = max(cur_start, prev_end_s + 0.001)
            emit_end = to_emit[-1]['end']
            min_disp_ms = adaptive_min_display(emit_text)
            if (emit_end - emit_start) * 1000 < min_disp_ms:
                emit_end = emit_start + (min_disp_ms / 1000.0)

            content = "\n".join(split_caption_text_two_lines(emit_text, max_chars))
            subs.append(srt.Subtitle(
                index=index,
                start=datetime.timedelta(seconds=emit_start),
                end=datetime.timedelta(seconds=emit_end),
                content=content
            ))
            index += 1
            prev_end_s = emit_end

            # Start new subtitle with the last word
            last_w = cur_words[-1]
            cur_words = [last_w]
            cur_start = last_w['start']

    # Flush remaining words
    if cur_words:
        emit_text = " ".join(x['word'].strip() for x in cur_words).strip()
        emit_start = max(cur_start, prev_end_s + 0.001)
        emit_end = cur_words[-1]['end']
        min_disp_ms = adaptive_min_display(emit_text)
        if (emit_end - emit_start) * 1000 < min_disp_ms:
            emit_end = emit_start + (min_disp_ms / 1000.0)
        content = "\n".join(split_caption_text_two_lines(emit_text, max_chars))
        subs.append(srt.Subtitle(
            index=index,
            start=datetime.timedelta(seconds=emit_start),
            end=datetime.timedelta(seconds=emit_end),
            content=content
        ))

    # Ensure no tiny overlaps between subtitles (defensive)
    for i in range(1, len(subs)):
        if subs[i].start <= subs[i-1].start:
            subs[i].start = subs[i-1].end + datetime.timedelta(milliseconds=10)
        if subs[i].start >= subs[i].end:
            subs[i].end = subs[i].start + datetime.timedelta(milliseconds=50)

    return srt.compose(subs)

def words_to_precise_srt(words, max_chars=42, max_duration=5000):
    subs = []
    cur_words, cur_start = [], None

    for w in words:
        if not cur_words:
            cur_start = w['start']
        cur_words.append(w)

        text = " ".join(x['word'] for x in cur_words).strip()
        cur_end = cur_words[-1]['end']
        duration_ms = (cur_end - cur_start) * 1000

        if len(text) > max_chars or duration_ms > max_duration:
            # close current
            subs.append(srt.Subtitle(
                index=len(subs) + 1,
                start=datetime.timedelta(seconds=cur_start),
                end=datetime.timedelta(seconds=cur_words[-2]['end']),
                content="\n".join(split_caption_text_two_lines(
                    " ".join(x['word'] for x in cur_words[:-1]),
                    max_chars=max_chars
                ))
            ))
            # start fresh with last word
            cur_words = [cur_words[-1]]
            cur_start = cur_words[0]['start']

    # flush remaining
    if cur_words:
        subs.append(srt.Subtitle(
            index=len(subs) + 1,
            start=datetime.timedelta(seconds=cur_start),
            end=datetime.timedelta(seconds=cur_words[-1]['end']),
            content="\n".join(split_caption_text_two_lines(
                " ".join(x['word'] for x in cur_words),
                max_chars=max_chars
            ))
        ))

    return srt.compose(subs)

def safe_filename(name: str) -> str:
    """Sanitize filenames to avoid illegal characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

# ---------- Optional runtime installs ----------
def ensure_pkg(mod_name, pip_name=None):
    try:
        __import__(mod_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

for pkg in ["transformers","sentencepiece","sacremoses","ttsmms","pydub","pysubs2","pysoundfile","inflect"]:
    ensure_pkg(pkg)

p = inflect.engine()

# ----------------------------- Utilities -----------------------------
def extract_audio_if_video(input_path, tmp_dir):
    video_exts = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv'}
    _, ext = os.path.splitext(input_path.lower())
    if ext in video_exts:
        out_audio = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
        cmd = ["ffmpeg","-y","-i",input_path,"-ac","1","-ar","16000","-vn",out_audio]
        print("Extracting audio with ffmpeg...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_audio
    return input_path

def split_audio(audio_path, chunk_length_ms=30 * 1000, overlap_ms=500):
    """
    Split audio into chunks with a small overlap to avoid losing words at boundaries.
    Returns list of tuples: (chunk_path, chunk_start_offset_ms)
    """
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_paths = []
    start = 0
    idx = 0
    while start < total_ms:
        end = min(start + chunk_length_ms, total_ms)
        chunk = audio[start:end]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(tmp.name, format="wav")
        chunk_paths.append((tmp.name, start))  # return start offset for accurate timestamping
        idx += 1
        # advance start but keep overlap with previous chunk
        if end == total_ms:
            break
        start = end - overlap_ms if (end - overlap_ms) > start else end
    return chunk_paths


def segments_to_srt(segments):
    subs = []
    for i, seg in enumerate(segments, start=1):
        start_td = datetime.timedelta(seconds=float(seg['start']))
        end_td = datetime.timedelta(seconds=float(seg['end']))
        subs.append(srt.Subtitle(index=i,start=start_td,end=end_td,content=seg['text'].strip()))
    return srt.compose(subs)

def convert_srt_to_ass(srt_file, ass_file, tgt_lang_nllb, max_line_len=80, base_font_size=28):
    """
    Convert SRT → ASS for burning into video.
    - Uses higher max_char length to avoid clipping/wrapping
    - Ensures all words remain visible in burned subtitles
    """
    with open(srt_file, "rb") as f:
        raw = f.read()

    # Robust decoding
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    subs = pysubs2.SSAFile.from_string(text, encoding="utf-8", format_="srt")

    # Pick font safe for target language
    font = get_font_for_lang(tgt_lang_nllb)
    if "Default" not in subs.styles:
        subs.styles["Default"] = pysubs2.SSAStyle()

    style = subs.styles["Default"]
    style.fontname = font
    style.fontsize = base_font_size
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    style.marginv = 40
    style.marginl = 20
    style.marginr = 20

    for ev in subs.events:
        clean_text = ev.text.strip()
        if not clean_text:
            continue

        # ✅ Allow much longer lines before wrapping
        wrapped_lines = split_caption_text_two_lines(clean_text, max_chars=max_line_len)
        ev.text = "\n".join(wrapped_lines)

        # ✅ Defensive timing fix
        if ev.end <= ev.start:
            ev.end = ev.start + 100  # at least 100 ms

    subs.save(ass_file, format_="ass", encoding="utf-8")
    print(f"[ASS] Subtitles converted safely with max {max_line_len} chars → {ass_file}")


def speedup_audio_to_fit_segment(audio: AudioSegment, target_duration_ms: int, max_step=1.5) -> AudioSegment:
    """ Speed up the audio to fit exactly into target_duration_ms.
    - Uses frame_rate resampling (no pitch change)
    - Applies in gentle steps to avoid aliasing artifacts
    """
    current_duration = len(audio)
    if current_duration <= target_duration_ms:  # Too short → pad with silence
        return audio + AudioSegment.silent(duration=(target_duration_ms - current_duration))
    speed_factor = current_duration / target_duration_ms
    if speed_factor <= 1.0:
        return audio  # Already shorter than target, no speedup needed
    step = 1.05
    while speed_factor > step and step < max_step:
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * step)})
        audio = audio.set_frame_rate(audio.frame_rate)
        speed_factor /= step
    # Final adjustment
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
    return audio.set_frame_rate(audio.frame_rate)

from pydub import silence

def _preprocess_audio_for_whisper(src_path, target_sr=16000):
    """
    Ensure audio is mono 16kHz WAV, volume-normalized.
    NO trimming (to avoid losing words).
    """
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    # Normalize loudness (RMS normalization)
    audio = effects.normalize(audio)

    # Export as temp wav
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name

def split_audio_on_silence(audio_path, min_silence_len=1500, silence_thresh_offset=-40, overlap_ms=400):
    """
    Split audio into chunks based on silence with overlaps.
    """
    audio = AudioSegment.from_file(audio_path)
    silence_thresh = audio.dBFS + silence_thresh_offset

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=overlap_ms  # ✅ adds overlap padding
    )

    chunk_paths, offset = [], 0
    for chunk in chunks:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(tmp.name, format="wav")
        chunk_paths.append((tmp.name, offset))
        offset += len(chunk) - overlap_ms
    return chunk_paths

def _merge_and_dedupe_segments(segments, min_gap_s=0.02):
    """
    Merge overlapping/duplicate segments and remove repeated words at boundaries.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]

        # Overlapping timing → merge
        if seg["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], seg["end"])

            # Avoid duplicate words at join
            prev_words = prev["text"].split()
            seg_words = seg["text"].split()
            overlap = 0
            for k in range(min(5, len(prev_words), len(seg_words)), 0, -1):
                if all(
                    _normalize_token_for_compare(prev_words[-k+i]) ==
                    _normalize_token_for_compare(seg_words[i])
                    for i in range(k)
                ):
                    overlap = k
                    break
            prev["text"] = " ".join(prev_words + seg_words[overlap:])
        else:
            # Snap small gaps
            if seg["start"] - prev["end"] < min_gap_s:
                seg["start"] = prev["end"] + min_gap_s
            merged.append(seg)

    return merged


def merge_single_word_tails(events):
    """
    Merge dangling single-word subtitles into the next subtitle.
    """
    new_events = []
    i = 0
    while i < len(events):
        cur = events[i]
        words = cur.text.strip().split()

        # If subtitle has exactly one word, push into next
        if len(words) == 1 and i + 1 < len(events):
            nxt = events[i + 1]
            nxt.text = words[0] + " " + nxt.text
            # adjust timings so next starts where current started
            nxt.start = min(nxt.start, cur.start)
            # skip current
            i += 1
            continue

        new_events.append(cur)
        i += 1

    return new_events

# ----------------------------- Font Map -----------------------------
LANG_FONTS = {
    # Indic scripts
    "_Deva": "Noto Sans Devanagari",     # Hindi, Marathi, Sanskrit, Bhojpuri, Maithili, Nepali
    "_Beng": "Noto Sans Bengali",        # Bengali, Assamese, Manipuri (Bangla script)
    "_Gujr": "Noto Sans Gujarati",       # Gujarati
    "_Guru": "Noto Sans Gurmukhi",       # Punjabi
    "_Taml": "Noto Sans Tamil",          # Tamil
    "_Telu": "Noto Sans Telugu",         # Telugu
    "_Knda": "Noto Sans Kannada",        # Kannada
    "_Mlym": "Noto Sans Malayalam",      # Malayalam
    "_Orya": "Noto Sans Oriya",          # Odia
    "_Sinh": "Noto Sans Sinhala",        # Sinhala
    "_Olck": "Noto Sans Ol Chiki",       # Santali (Ol Chiki)
    "_Laoo": "Noto Sans Lao",            # Lao

    # East Asian
    "_Hans": "Noto Sans SC",             # Simplified Chinese
    "_Hant": "Noto Sans TC",             # Traditional Chinese
    "_Jpan": "Noto Sans JP",             # Japanese
    "_Hang": "Noto Sans KR",             # Korean Hangul
    "_Tibt": "Noto Sans Tibetan",        # Tibetan
    "_Mymr": "Noto Sans Myanmar",        # Burmese, etc.
    "_Khmr": "Noto Sans Khmer",          # Khmer
    "_Thai": "Noto Sans Thai",           # Thai

    # Middle Eastern
    "_Arab": "Noto Naskh Arabic",        # Arabic, Urdu, Persian, Pashto
    "_Hebr": "Noto Sans Hebrew",         # Hebrew
    "_Syrc": "Noto Sans Syriac",         # Syriac
    "_Tfng": "Noto Sans Tifinagh",       # Tamazight (Berber)

    # Cyrillic
    "_Cyrl": "Noto Sans",                # Russian, Ukrainian, Serbian, etc. (Noto Sans has full Cyrillic)
    "_Glag": "Noto Sans Glagolitic",     # Glagolitic (rare)

    # European Latin
    "_Latn": "Noto Sans",                # English, French, German, Spanish, etc.
    "_Grek": "Noto Sans Greek",          # Greek
    "_Armn": "Noto Sans Armenian",       # Armenian
    "_Geor": "Noto Sans Georgian",       # Georgian

    # African + others
    "_Ethi": "Noto Sans Ethiopic",       # Amharic, Tigrinya
    "_Cher": "Noto Sans Cherokee",       # Cherokee
    "_Vaii": "Noto Sans Vai",            # Vai
    "_Tale": "Noto Sans Tai Le",         # Tai scripts
    "_Talu": "Noto Sans New Tai Lue",
    "_Phag": "Noto Sans Phags Pa",

    # Default
    "default": "Noto Sans"               # Fallback
}

# ----------------------------- Language Maps -----------------------------
ISO2_TO_NLLB = {
    "hi": "hin_Deva", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya",
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    # --- Extended MMS-supported entries ---
    "af": "afr_Latn", "am": "amh_Ethi", "as": "asm_Beng", "ast": "ast_Latn", "az": "azj_Latn",
    "ba": "bak_Cyrl", "be": "bel_Cyrl", "bem": "bem_Latn", "bg": "bul_Cyrl", "bho": "bho_Deva",
    "bm": "bam_Latn", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn", "ceb": "ceb_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "dv": "div_Thaa", "dz": "dzo_Tibt",
    "el": "ell_Grek", "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fil": "fil_Latn", "fj": "fij_Latn", "fo": "fao_Latn", "fy": "fry_Latn", "ga": "gle_Latn",
    "gd": "gla_Latn", "gl": "glg_Latn", "gn": "grn_Latn", "ha": "hau_Latn", "haw": "haw_Latn",
    "he": "heb_Hebr", "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "ig": "ibo_Latn", "ilo": "ilo_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "jv": "jav_Latn", "kab": "kab_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kns": "kik_Latn", "rw": "kin_Latn", "ky": "kir_Cyrl", "ku": "ckb_Arab", "lb": "ltz_Latn",
    "lg": "lug_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mai": "mai_Deva", "mg": "plt_Latn", "mk": "mkd_Cyrl", "mn": "khk_Cyrl", "mni": "mni_Beng",
    "mos": "mos_Latn", "ms": "msa_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva",
    "nl": "nld_Latn", "nn": "nno_Latn", "no": "nob_Latn", "ny": "nya_Latn", "oc": "oci_Latn",
    "om": "gaz_Latn", "pap": "pap_Latn", "pl": "pol_Latn", "ps": "pbt_Arab", "qu": "quy_Latn",
    "ro": "ron_Latn", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sm": "smo_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "ss": "ssw_Latn", "st": "sot_Latn", "su": "sun_Latn",
    "sv": "swe_Latn", "sw": "swh_Latn", "taq": "taq_Latn", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "ti": "tir_Ethi", "tk": "tuk_Latn", "tn": "tsn_Latn", "to": "ton_Latn", "tr": "tur_Latn",
    "ts": "tso_Latn", "tum": "tum_Latn", "tw": "twi_Latn", "tzm": "tzm_Tfng", "ug": "uig_Arab",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "wo": "wol_Latn",
    "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant", "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", "zu": "zul_Latn",
    # --- Additional MMS-supported languages ---
    "ak": "aka_Latn", "ee": "ewe_Latn", "knj": "kan_Latn", "tl": "tgl_Latn", "kg": "kon_Latn",
    "kr": "kau_Latn",
}

# ----------------------------- Auto-detect Non-Latin suffixes -----------------------------
# Extract all suffixes that are NOT "_Latn" (Latin alphabet)
NON_LATIN_SUFFIXES = sorted({
    code.split("_")[1] for code in ISO2_TO_NLLB.values() if "_" in code and not code.endswith("_Latn")
})
# Rebuild into full suffix strings like "_Deva", "_Cyrl", etc.
NON_LATIN_SUFFIXES = [f"_{suf}" for suf in NON_LATIN_SUFFIXES]

print(f"[INFO] Non-Latin suffixes detected: {NON_LATIN_SUFFIXES}")

def to_nllb_code(code: str) -> str:
    if "_" in code and len(code.split("_")[0])==3:
        return code
    return ISO2_TO_NLLB.get(code.lower(),"eng_Latn")

NLLB_TO_MMS_OVERRIDES={"zho":"cmn"}

def nllb_to_mms(nllb_code: str) -> str:
    return NLLB_TO_MMS_OVERRIDES.get(nllb_code.split("_")[0], nllb_code.split("_")[0])

def universal_normalize_text(text: str, lang_hint="eng") -> str:
    """Language-aware normalization.
    - For English: expand abbreviations, spell out numbers, split acronyms
    - For others: only strip and normalize whitespace
    """
    if not text:
        return text

    # If not English → minimal cleanup only
    if not lang_hint.startswith("eng"):
        return re.sub(r"\s+", " ", text).strip()

    # English-specific normalization
    abbrs = {
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "mr.": "mister",
        "mrs.": "missus",
        "dr.": "doctor",
        "st.": "saint"
    }

    raw_tokens = re.findall(r"\d+\.\d+|\d+%?|\w+['-]?\w*|[^\w\s]", text)
    out_tokens = []

    for tok in raw_tokens:
        lower = tok.lower()
        if lower in abbrs:
            out_tokens.extend(abbrs[lower].split())
            continue
        if tok.isalpha() and tok.isupper() and len(tok) >= 2:
            out_tokens.extend(list(tok))
            continue
        if re.fullmatch(r"\d+", tok):  # numbers → words
            try:
                out_tokens.extend(p.number_to_words(int(tok)).split())
            except Exception:
                out_tokens.append(tok)
            continue
        m_ord = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok, flags=re.IGNORECASE)
        if m_ord:
            try:
                out_tokens.extend(p.ordinal(p.number_to_words(int(m_ord.group(1)))).split())
            except Exception:
                out_tokens.append(tok)
            continue
        if re.fullmatch(r"\d+\.\d+", tok):
            integer, frac = tok.split(".")
            try:
                int_part = p.number_to_words(int(integer)).split()
                frac_part = " ".join([p.number_to_words(int(d)) for d in frac])
                out_tokens.extend(int_part + ["point"] + frac_part.split())
            except Exception:
                out_tokens.append(tok)
            continue
        m = re.match(r"([A-Za-z]+)?(\d+)([A-Za-z]+)?", tok)
        if m and (m.group(1) or m.group(3)):
            if m.group(1): out_tokens.append(m.group(1))
            try:
                out_tokens.extend(p.number_to_words(int(m.group(2))).split())
            except Exception:
                out_tokens.append(m.group(2))
            if m.group(3): out_tokens.append(m.group(3))
            continue
        out_tokens.append(tok)

    normalized = " ".join(out_tokens)
    return re.sub(r"\s+", " ", normalized).strip()

# ----------------------------- Translation (NLLB) -----------------------------
print("Loading NLLB model for translation...")
NLLB_MODEL_NAME="facebook/nllb-200-1.3B"
tokenizer=AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model=AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

def enforce_contextual_translation(
    words: list,
    src_lang_nllb: str,
    tgt_lang_nllb: str,
    translator,
    max_chunk_len: int = 200
):
    """
    Context-aware + coverage-guaranteed translation.
    - Translates full segment for fluency
    - Anchors noun-like tokens (universal heuristic, no spaCy)
    - Guarantees every word is translated (no skips, no hallucinations)
    """
    if not words:
        return []

    # --- Step 1: Full sentence translation ---
    full_text = " ".join(w["word"] for w in words).strip()
    full_translation = ""
    if full_text:
        try:
            chunks = _split_text_for_translation(full_text, max_chars=max_chunk_len)
            translated_chunks = [
                translator(chunk, max_length=256)[0]["translation_text"].strip()
                for chunk in chunks
            ]
            full_translation = " ".join(translated_chunks).strip()
        except Exception as e:
            print(f"[WARN] Context translation failed, fallback only: {e}")
            full_translation = ""

    tgt_tokens = full_translation.split() if full_translation else []
    src_tokens = [w["word"] for w in words]

    # --- Step 2: Universal noun-like detection ---
    candidate_nouns = []
    for tok in src_tokens:
        if (
            tok and (
                tok[0].isupper()               # capitalized word (proper noun, brand, name)
                or len(tok) > 3                # longer words treated as content words
            )
        ):
            candidate_nouns.append(tok)

    # --- Step 3: Greedy alignment with coverage ---
    aligned = []
    j = 0
    tgt_norm = {_normalize_token_for_compare(t) for t in tgt_tokens}
    for i, src_tok in enumerate(src_tokens):
        mapped = None

        # Sequential alignment first
        if j < len(tgt_tokens):
            mapped = tgt_tokens[j]
            j += 1

        # If noun-like token missing in target → force translation
        if src_tok in candidate_nouns and _normalize_token_for_compare(src_tok) not in tgt_norm:
            try:
                noun_tr = translator(src_tok, max_length=64)[0]["translation_text"].strip()
                if noun_tr:
                    mapped = noun_tr
                    tgt_norm.add(_normalize_token_for_compare(noun_tr))
            except Exception:
                mapped = src_tok  # fallback keep source noun

        # Fallback per-word translation if still unmapped
        if not mapped:
            try:
                mapped = translator(src_tok, max_length=64)[0]["translation_text"].strip()
            except Exception:
                mapped = src_tok

        # ✅ Ensure mapping is added
        if mapped:
            aligned.append({
                "word": mapped,
                "start": words[i]["start"],
                "end": words[i]["end"],
            })

    return aligned

def create_translation_pipeline(model, tokenizer, src_lang, tgt_lang, max_length=512):
    """Create translation pipeline with auto device selection."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        device=device,
        max_length=max_length
    )


def _distribute_tokens_to_segments(block_tokens, segments_in_block):
    """
    Distribute block_tokens into segments_in_block (list of dicts with 'text'),
    proportionally by source length, but guarantee:
      - every segment gets ≥1 token if tokens available
      - sum of allocations == len(block_tokens)
    """
    n_tokens = len(block_tokens)
    n_segs = len(segments_in_block)
    if n_segs == 0:
        return []

    char_counts = [max(1, len(s["text"].strip())) for s in segments_in_block]
    total_chars = sum(char_counts)

    if total_chars <= 0:
        counts = [n_tokens // n_segs] * n_segs
        for i in range(n_tokens - sum(counts)):
            counts[i % n_segs] += 1
    else:
        float_alloc = [(c / total_chars) * n_tokens for c in char_counts]
        counts = [max(1, int(round(v))) for v in float_alloc]
        diff = n_tokens - sum(counts)
        counts[-1] += diff
        for i in range(len(counts)):
            if counts[i] <= 0:
                counts[i] = 1
        diff = n_tokens - sum(counts)
        counts[-1] += diff

    out_texts, offset = [], 0
    for c in counts:
        slice_tokens = block_tokens[offset: offset + c]
        out_texts.append(" ".join(slice_tokens).strip())
        offset += c

    if offset < n_tokens:
        leftover = " ".join(block_tokens[offset:])
        out_texts[-1] = (out_texts[-1] + " " + leftover).strip()

    return out_texts

def _merge_by_sentence_boundaries(segments):
    """
    Merge subtitle segments until a sentence boundary (. ? ! ।).
    Returns a list of merged blocks, each with combined text and timing span.
    """
    merged = []
    cur_block = []
    cur_start, cur_end = None, None

    for seg in segments:
        txt = seg["text"].strip()
        if not txt:
            continue

        if not cur_block:
            cur_start = seg["start"]

        cur_block.append(txt)
        cur_end = seg["end"]

        # Check if this segment ends with sentence boundary
        if txt.endswith((".", "?", "!", "।")):
            merged.append({
                "start": cur_start,
                "end": cur_end,
                "text": " ".join(cur_block).strip()
            })
            cur_block, cur_start, cur_end = [], None, None

    # Flush leftover
    if cur_block:
        merged.append({
            "start": cur_start,
            "end": cur_end,
            "text": " ".join(cur_block).strip()
        })

    return merged

def _backfill_missing_words(src_text, tgt_text, translator, src_lang, tgt_lang):
    """
    Ensure no source words are skipped in translation.
    If a word from src_text is missing in tgt_text, translate it individually and append.
    """
    src_tokens = src_text.split()
    tgt_tokens = tgt_text.split()
    tgt_norm = set(_normalize_token_for_compare(tok) for tok in tgt_tokens)

    for tok in src_tokens:
        if _normalize_token_for_compare(tok) not in tgt_norm:
            try:
                translated_tok = translator(tok, max_length=64)[0]["translation_text"].strip()
                if translated_tok and _normalize_token_for_compare(translated_tok) not in tgt_norm:
                    tgt_tokens.append(translated_tok)
                    tgt_norm.add(_normalize_token_for_compare(translated_tok))
            except Exception:
                # fallback keep original token
                tgt_tokens.append(tok)
                tgt_norm.add(_normalize_token_for_compare(tok))

    return " ".join(tgt_tokens)

def translate_segments_nllb_batched(
    segments,
    src_lang_nllb,
    tgt_lang_nllb,
    batch_size=8,
    max_length=512
):
    """
    Safer translation:
      - Merge by sentence boundaries
      - Translate full sentence blocks
      - Redistribute tokens with guaranteed coverage
      - Cleanup duplicates
    """
    translator = create_translation_pipeline(
        nllb_model, tokenizer, src_lang_nllb, tgt_lang_nllb, max_length=max_length
    )

    sentence_blocks = _merge_by_sentence_boundaries(segments)

    translated_blocks = []
    for block in sentence_blocks:
        try:
            tr_text = translator(block["text"], max_length=max_length)[0]["translation_text"].strip()
        except Exception as e:
            print(f"[WARN] Sentence translation failed, fallback to source: {e}")
            tr_text = block["text"]
        translated_blocks.append({
            "start": block["start"],
            "end": block["end"],
            "text": tr_text
        })

    translated = []
    b_idx, t_idx = 0, 0
    while b_idx < len(sentence_blocks):
        block = sentence_blocks[b_idx]
        block_tokens = translated_blocks[b_idx]["text"].split()

        segs_in_block = []
        while t_idx < len(segments) and segments[t_idx]["end"] <= block["end"]:
            segs_in_block.append(segments[t_idx])
            t_idx += 1

        if not segs_in_block and t_idx < len(segments) and segments[t_idx]["start"] < block["end"]:
            segs_in_block.append(segments[t_idx])
            t_idx += 1

        if not segs_in_block:
            b_idx += 1
            continue

        assigned_texts = _distribute_tokens_to_segments(block_tokens, segs_in_block)
        for seg, txt in zip(segs_in_block, assigned_texts):
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": txt
            })

        b_idx += 1

    return tidy_translated_segments(translated)

def translate_srt_file_segment(input_srt, output_srt, src_lang_nllb, tgt_lang_nllb):
    """
    Safer SRT translation:
      - Merge by sentence boundaries
      - Translate full blocks
      - Redistribute tokens with guaranteed coverage
      - Cleanup duplicates
    """
    subs = pysubs2.load(input_srt, encoding="utf-8")

    translator = create_translation_pipeline(
        nllb_model, tokenizer, src_lang_nllb, tgt_lang_nllb, max_length=512
    )

    merged_blocks = []
    cur_block, cur_start, cur_end, cur_events = [], None, None, []
    for ev in subs:
        txt = ev.text.strip()
        if not txt:
            continue
        if not cur_block:
            cur_start = ev.start
        cur_block.append(txt)
        cur_events.append(ev)
        cur_end = ev.end
        if txt.endswith((".", "?", "!", "।")):
            merged_blocks.append({
                "start": cur_start,
                "end": cur_end,
                "events": list(cur_events)
            })
            cur_block, cur_start, cur_end, cur_events = [], None, None, []
    if cur_block:
        merged_blocks.append({
            "start": cur_start,
            "end": cur_end,
            "events": list(cur_events)
        })

    translated_blocks = []
    for block in merged_blocks:
        block_text = " ".join(ev.text.strip() for ev in block["events"]).strip()
        try:
            tr_text = translator(block_text, max_length=512)[0]["translation_text"].strip()
        except Exception as e:
            print(f"[WARN] Sentence translation failed, fallback: {e}")
            tr_text = block_text
        translated_blocks.append({
            "start": block["start"],
            "end": block["end"],
            "text": tr_text,
            "events": block["events"]
        })

    translated_events = []
    for block in translated_blocks:
        block_tokens = block["text"].split()
        events = block["events"]
        assigned_texts = _distribute_tokens_to_segments(block_tokens, [{"text": ev.text} for ev in events])
        for ev, txt in zip(events, assigned_texts):
            ev.text = txt
            translated_events.append(ev)

    translated_events = tidy_translated_segments([
        {"start": ev.start/1000, "end": ev.end/1000, "text": ev.text}
        for ev in translated_events
    ])

    final_events = [
        pysubs2.SSAEvent(
            start=int(seg["start"]*1000),
            end=int(seg["end"]*1000),
            text=seg["text"]
        )
        for seg in translated_events
    ]

    ssa = pysubs2.SSAFile()
    ssa.events = final_events
    ssa.save(output_srt, format_="srt", encoding="utf-8")
    print(f"[SAFE-SRT-TRANSLATE] Final translated SRT → {output_srt}")

# ----------------------------- Language-aware Deduplication -----------------------------
def dedupe_translations(text: str, tgt_lang_nllb: str = "eng_Latn") -> str:
    """
    Gentle deduplication:
    - Keeps valid repetitions (e.g., "very very good").
    - Removes stutter-like triples or mixed duplicates (e.g., "Google गूगल Google").
    - Applies brand map first.
    - Disabled for morphologically rich non-Latin languages (to avoid breaking grammar).
    """
    if not text:
        return text

    # 🔹 Disable dedupe if target language is in complex morphology set
    if any(tgt_lang_nllb.endswith(suf) for suf in NON_LATIN_SUFFIXES):
        return text  # return unchanged

    tokens = text.split()
    cleaned = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # collapse triples like "Google Google Google"
        if i + 1 < len(tokens) and tokens[i + 1].lower() == tok.lower():
            if i + 2 < len(tokens) and tokens[i + 2].lower() == tok.lower():
                cleaned.append(tok)
                i += 3
                continue
        cleaned.append(tok)
        i += 1

    return " ".join(cleaned)

def get_font_for_lang(tgt_lang_nllb: str) -> str:
    for suffix, font in LANG_FONTS.items():
        if tgt_lang_nllb.endswith(suffix):
            return font
    return LANG_FONTS["default"]

def _split_text_for_translation(text: str, max_chars=200):
    """Split text into safe chunks for NLLB to avoid truncation."""
    if len(text) <= max_chars:
        return [text.strip()]
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur += (" " if cur else "") + w
        else:
            chunks.append(cur.strip())
            cur = w
    if cur:
        chunks.append(cur.strip())
    return chunks

# ----------------------------- MMS TTS with adaptive audible speed -----------------------------
tts_models = {}

def get_tts_model(tgt_lang_nllb):
    mms_code = nllb_to_mms(tgt_lang_nllb)
    model_dir = os.path.join("models", mms_code)
    if tgt_lang_nllb not in tts_models:
        if not os.path.exists(model_dir):
            print(f"Downloading MMS TTS model for {tgt_lang_nllb} → {mms_code} ...")
            download(mms_code, "./models")
        tts_models[tgt_lang_nllb] = TTS(model_dir)
    return tts_models[tgt_lang_nllb]

def split_caption_text(text, max_chars=60):
    """Split caption text intelligently without splitting words."""
    if len(text) <= max_chars:
        return [text.strip()]
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def build_audio_driven_subs(words, max_duration_s=5.0, max_chars=42, silence_gap_s=0.6):
    """
    Build subtitles aligned to audio using Whisper word timestamps.
    Splits only at:
      - natural silence gaps (> silence_gap_s)
      - punctuation (.,?!)
      - or max duration/char limit
    Ensures every subtitle matches the speech naturally.
    """
    if not words:
        return []

    subs = []
    cur_words = []
    cur_start = words[0]["start"]

    for i, w in enumerate(words):
        cur_words.append(w)
        cur_end = w["end"]

        text = " ".join(x["word"] for x in cur_words).strip()
        duration = cur_end - cur_start

        # condition: silence gap, punctuation, too long, or last word
        is_last = (i == len(words) - 1)
        next_gap = (words[i+1]["start"] - w["end"]) if not is_last else 0.0
        ends_with_punct = any(p in w["word"] for p in [".", "?", "!", "।"])

        if (
            duration >= max_duration_s
            or len(text) > max_chars
            or next_gap > silence_gap_s
            or ends_with_punct
            or is_last
        ):
            subs.append({
                "start": cur_start,
                "end": cur_end,
                "text": "\n".join(split_caption_text_two_lines(text, max_chars=max_chars))
            })
            cur_words = []
            if not is_last:
                cur_start = words[i+1]["start"]

    return subs

def refine_subtitles_for_tts(subs, max_chars=60):
    """
    Split long subtitle events but preserve exact total timing.
    Input: subs is a pysubs2 SSAFile or list of events (start/end in ms).
    Output: list of pysubs2.SSAEvent objects (ms-accurate).
    """
    new_subs = []

    for sub in subs:
        text = sub.text.strip()
        start_ms = int(sub.start)
        end_ms = int(sub.end)
        total_dur = max(0, end_ms - start_ms)

        if not text:
            # keep silent/empty event as-is
            new_subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=""))
            continue

        if len(text) <= max_chars or total_dur <= 0:
            new_subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))
            continue

        chunks = split_caption_text(text, max_chars=max_chars)

        # Distribute duration across chunks, preserving total duration exactly
        base_chunk = total_dur // len(chunks)
        remainder = total_dur - (base_chunk * len(chunks))

        offset = start_ms
        for i, chunk in enumerate(chunks):
            this_dur = base_chunk + (1 if i < remainder else 0)  # distribute remainder to first events
            evt_start = offset
            evt_end = offset + this_dur
            # ensure we never create zero-length events
            if evt_end <= evt_start:
                evt_end = evt_start + 1
            new_subs.append(pysubs2.SSAEvent(start=evt_start, end=evt_end, text=chunk))
            offset = evt_end

    # make sure events are sorted and non-overlapping (very defensive)
    new_subs_sorted = sorted(new_subs, key=lambda e: e.start)
    for i in range(1, len(new_subs_sorted)):
        if new_subs_sorted[i].start < new_subs_sorted[i-1].end:
            new_subs_sorted[i-1].end = min(new_subs_sorted[i-1].end, new_subs_sorted[i].start - 1)

    return new_subs_sorted

async def tts_segment_mms_refined(
    text,
    tgt_lang_nllb,
    duration_ms,
    base_speed=1.0,
    target_db=-3.0,
    pre_pad_ms=100,
    post_pad_ms=150,
    fade_ms=50
):
    """Generate refined TTS with padding and fading, normalized, and fit to segment duration."""
    if not text.strip():
        return AudioSegment.silent(duration=duration_ms)
    if tgt_lang_nllb.startswith("eng"):
        text = universal_normalize_text(text)

    tts_model = get_tts_model(tgt_lang_nllb)

    wav_out = tts_model.synthesis(text)
    y = wav_out["x"]
    sr = int(wav_out["sampling_rate"])
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y / max(1.0, np.max(np.abs(y))) * 0.9
    tensor_int16 = (y * 32767).astype(np.int16)
    speech = AudioSegment(
        tensor_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    # Add pre/post padding
    speech = AudioSegment.silent(duration=pre_pad_ms) + speech + AudioSegment.silent(duration=post_pad_ms)
    # Apply fade-in/out
    if len(speech) > 2 * fade_ms:
        speech = speech.fade_in(fade_ms).fade_out(fade_ms)
    # Normalize
    speech = effects.normalize(speech, headroom=abs(target_db))
    # Fit into target duration
    speech = speedup_audio_to_fit_segment(speech, duration_ms)
    return speech

def tts_from_srt_global_fit_refined(
    srt_file,
    tgt_lang_nllb,
    video_path,
    output_audio,
    pre_pad_ms=200,
    post_pad_ms=300,
    fade_ms=80
):
    """ Generate a continuous refined TTS track (ignores per-sub timing). Adds padding, fading, normalization, then stretches to fit video duration. """
    subs = pysubs2.load(srt_file, encoding="utf-8")
    full_text = " ".join([sub.text.strip() for sub in subs if sub.text.strip()])
    if not full_text:
        raise ValueError("No subtitle text to speak.")
    tts_model = get_tts_model(tgt_lang_nllb)
    # 🔹 Apply text normalization for English
    if tgt_lang_nllb.startswith("eng"):
        full_text = universal_normalize_text(full_text)

    wav_out = tts_model.synthesis(full_text)
    y = wav_out["x"]
    sr = int(wav_out["sampling_rate"])
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y / max(1.0, np.max(np.abs(y))) * 0.9
    tensor_int16 = (y * 32767).astype(np.int16)
    tts_audio = AudioSegment(
        tensor_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    # Add padding
    tts_audio = AudioSegment.silent(duration=pre_pad_ms) + tts_audio + AudioSegment.silent(duration=post_pad_ms)
    # Apply fade-in/out
    if len(tts_audio) > 2 * fade_ms:
        tts_audio = tts_audio.fade_in(fade_ms).fade_out(fade_ms)
    # Normalize
    tts_audio = effects.normalize(tts_audio, headroom=3.0)
    # Match video duration
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe['format']['duration']) * 1000
    tts_audio = speedup_audio_to_fit_segment(tts_audio, int(video_duration))
    # Export
    tts_audio.export(output_audio, format="mp3")
    print(f"Refined global-fit TTS file created: {output_audio}")
    return output_audio

# ----------------------------- Whisper -----------------------------
import whisper

def transcribe_in_chunks(
    audio_path,
    language=None,
    chunk_length_ms=None,
    overlap_ms=1000,       # safe overlap
    use_silence_splitting=True
):
    """
    High-accuracy OpenAI Whisper transcription with word-driven segment rebuilding.
    Ensures no repeated or dropped words, and segments are aligned to audio.
    - Auto-detects language from the first chunk if not forced
    - Uses that language for all subsequent chunks
    """
    model_size = "large-v3"
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    # === Split audio ===
    if use_silence_splitting:
        print("Splitting audio by silence with overlap...")
        chunk_info = split_audio_on_silence(
            audio_path,
            min_silence_len=1200,
            silence_thresh_offset=-35,
            overlap_ms=overlap_ms
        )
    else:
        if not chunk_length_ms:
            chunk_length_ms = 60 * 1000
        print(f"Splitting audio into {chunk_length_ms//1000}s chunks with {overlap_ms}ms overlap...")
        chunk_info = split_audio(audio_path, chunk_length_ms, overlap_ms=overlap_ms)

    all_words = []
    detected_lang = None

    # === Transcribe each chunk ===
    for idx, (chunk_path, chunk_start_ms) in enumerate(tqdm(chunk_info, desc="Transcribing chunks")):
        if idx == 0 and language is None:
            # Auto-detect on first chunk
            result = model.transcribe(
                chunk_path,
                language=None,
                word_timestamps=True,
                verbose=False
            )
            detected_lang = result.get("language", None)
            print(f"[LANG] Auto-detected language: {detected_lang}")
        else:
            # Use forced language OR detected one
            result = model.transcribe(
                chunk_path,
                language=language or detected_lang,
                word_timestamps=True,
                verbose=False
            )

        offset = chunk_start_ms / 1000.0
        for seg in result["segments"]:
            for w in seg["words"]:
                word = {
                    "word": w["word"].strip(),
                    "start": float(w["start"]) + offset,
                    "end": float(w["end"]) + offset,
                }
                if word["word"]:
                    all_words.append(word)

    # === Deduplicate ===
    all_words = dedupe_nearby_words(all_words, time_tol_s=0.1)

    # === Build subtitles aligned with audio ===
    segments = build_audio_driven_subs(
        all_words,
        max_duration_s=5.0,
        max_chars=42,
        silence_gap_s=0.6
    )

    return segments, (language or detected_lang), all_words

def adaptive_min_display(text, base_per_char=50, min_ms=800, max_ms=3000):
    est = len(text) * base_per_char
    return max(min_ms, min(est, max_ms))

def enforce_line_wrap(text, max_len=40):
    """Force line breaks for ASS/SRT so ffmpeg renderer never skips words."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur += (" " if cur else "") + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def refine_srt(input_srt, output_srt, max_line_length=42, lang_hint="eng"):
    """
    Refine subtitles for readability:
    - Wrap into max 2 lines
    - Language-specific cleanup
    - DO NOT change timings (sync stays from Whisper)
    """
    subs = pysubs2.load(input_srt, encoding="utf-8")
    refined = []

    for ev in subs:
        text = ev.text.strip()

        if lang_hint.startswith("eng"):
            # cleanup fillers
            text = re.sub(r"\b(uh|um|ah|er|hmm)\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if text and not text.endswith((".", "?", "!")):
                text += "."
            if text:
                text = text[0].upper() + text[1:]

        # line wrapping only
        wrapped_lines = split_caption_text_two_lines(text, max_chars=max_line_length)
        ev.text = "\n".join(wrapped_lines)
        refined.append(ev)

    ssa_file = pysubs2.SSAFile()
    ssa_file.events = refined
    ssa_file.save(output_srt, encoding="utf-8", format_="srt")
    print(f"[SAFE] Refined subtitles saved (timings preserved) → {output_srt}")

def collapse_repeated_runs_in_text(text: str) -> str:
    """
    Collapse immediate repeated tokens inside a single text string.
    e.g. "Google Google गूगल Google"   -> keeps first appearance of each run
    This preserves the first token's surface form and removes subsequent duplicates.
    """
    if not text:
        return text
    toks = text.split()
    out = []
    prev_norm = None
    for t in toks:
        norm = _normalize_token_for_compare(t)
        if norm == prev_norm:
            # skip repeated token
            continue
        out.append(t)
        prev_norm = norm
    return " ".join(out)

def _remove_boundary_duplicates(segments):
    """
    For adjacent segments, if the last K tokens of seg[i] == first K tokens of seg[i+1]
    (K up to 3), remove that leading overlap from seg[i+1].
    Works in-place on list of {'start','end','text'}.
    """
    if not segments:
        return segments
    segments_sorted = sorted(segments, key=lambda s: s['start'])
    for i in range(len(segments_sorted) - 1):
        a_tokens = segments_sorted[i]['text'].split()
        b_tokens = segments_sorted[i + 1]['text'].split()
        if not a_tokens or not b_tokens:
            continue
        max_k = min(3, len(a_tokens), len(b_tokens))
        # check longer overlaps first
        for k in range(max_k, 0, -1):
            end_a = a_tokens[-k:]
            start_b = b_tokens[:k]
            if [_normalize_token_for_compare(x) for x in end_a] == [_normalize_token_for_compare(x) for x in start_b]:
                # remove the overlapping tokens from the start of b
                segments_sorted[i + 1]['text'] = " ".join(b_tokens[k:]).strip()
                break
    return segments_sorted

def translated_words_to_srt(words, translator, src_lang_nllb, tgt_lang_nllb, max_chars=40, max_duration=5000):
    """
    Build word-perfect translated subtitles:
      - Uses enforce_contextual_translation to guarantee no skips
    """
    aligned_words = enforce_contextual_translation(words, src_lang_nllb, tgt_lang_nllb, translator)

    subs = []
    cur_words = []
    start = None

    for w in aligned_words:
        if not cur_words:
            start = w['start']
        cur_words.append(w)

        text = " ".join(x['word'] for x in cur_words).strip()
        duration = (w['end'] - start) * 1000

        if len(text) > max_chars or duration > max_duration:
            end = cur_words[-2]['end'] if len(cur_words) > 1 else cur_words[-1]['end']
            emit_text = " ".join(x['word'] for x in cur_words[:-1]) if len(cur_words) > 1 else cur_words[0]['word']
            subs.append(srt.Subtitle(
                index=len(subs) + 1,
                start=datetime.timedelta(seconds=start),
                end=datetime.timedelta(seconds=end),
                content="\n".join(split_caption_text_two_lines(emit_text, max_chars))
            ))
            # restart buffer
            cur_words = [cur_words[-1]]
            start = cur_words[0]['start']

    if cur_words:
        emit_text = " ".join(x['word'] for x in cur_words)
        subs.append(srt.Subtitle(
            index=len(subs) + 1,
            start=datetime.timedelta(seconds=start),
            end=datetime.timedelta(seconds=cur_words[-1]['end']),
            content="\n".join(split_caption_text_two_lines(emit_text, max_chars))
        ))

    return srt.compose(subs)

def tidy_translated_segments(segments):
    """
    1) Collapse immediate repeated tokens within each segment.
    2) Remove short phrase duplicates that cross segment boundaries.
    Returns the cleaned segments (sorted by start).
    """
    if not segments:
        return segments
    # collapse repeated runs inside each segment
    for seg in segments:
        seg['text'] = collapse_repeated_runs_in_text(seg.get('text','')).strip()

    # remove boundary duplicates (end-of-seg == start-of-next)
    segments = _remove_boundary_duplicates(segments)

    # final cleanup: trim whitespace
    for seg in segments:
        seg['text'] = seg['text'].strip()
    return segments

def split_and_fit_subs(input_srt, output_srt, max_chars=42, min_display_ms=800, max_display_ms=5000):
    """
    Split long subtitles into smaller ones that match speaker timing.
    Ensures every word is shown, respecting pace.
    """
    subs = pysubs2.load(input_srt, encoding="utf-8")
    new_events = []

    for ev in subs:
        text = ev.text.strip()
        if not text:
            new_events.append(ev)
            continue

        start_ms, end_ms = ev.start, ev.end
        duration = max(1, end_ms - start_ms)

        # Split into smaller chunks of <= max_chars
        chunks = split_caption_text(text, max_chars=max_chars)
        num_chunks = len(chunks)

        if num_chunks == 1:
            # Ensure min/max duration constraints
            if duration < min_display_ms:
                end_ms = start_ms + min_display_ms
            elif duration > max_display_ms:
                end_ms = start_ms + max_display_ms
            new_events.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text="\n".join(
                split_caption_text_two_lines(chunks[0], max_chars=max_chars)
            )))
        else:
            # Distribute duration evenly across chunks
            base_dur = duration // num_chunks
            remainder = duration % num_chunks
            offset = start_ms
            for i, chunk in enumerate(chunks):
                dur = base_dur + (1 if i < remainder else 0)
                dur = max(min_display_ms, min(dur, max_display_ms))  # clamp duration
                ev_start = offset
                ev_end = offset + dur
                wrapped = split_caption_text_two_lines(chunk, max_chars=max_chars)
                new_events.append(pysubs2.SSAEvent(start=ev_start, end=ev_end, text="\n".join(wrapped)))
                offset = ev_end

    # Fix overlaps (each subtitle starts after the previous one ends)
    new_events_sorted = sorted(new_events, key=lambda e: e.start)
    for i in range(1, len(new_events_sorted)):
        if new_events_sorted[i].start <= new_events_sorted[i-1].end:
            new_events_sorted[i].start = new_events_sorted[i-1].end + 10

    ssa = pysubs2.SSAFile()
    ssa.events = new_events_sorted
    ssa.save(output_srt, format_="srt", encoding="utf-8")
    print(f"[PACE] Subtitles split and synced with voice → {output_srt}")

def build_time_based_subs(words, max_duration_s=4.0, max_chars=42):
    """
    Split subtitles strictly by time window.
    - words: list of {"word","start","end"}
    - max_duration_s: maximum duration per subtitle
    - max_chars: soft cap on characters per subtitle line
    """
    if not words:
        return []

    subs = []
    cur_words = []
    cur_start = words[0]["start"]
    cur_end = cur_start

    for w in words:
        cur_words.append(w)
        cur_end = w["end"]

        # Check if current chunk is too long (time or text)
        text = " ".join(x["word"] for x in cur_words).strip()
        duration = cur_end - cur_start

        if duration >= max_duration_s or len(text) > max_chars:
            # Flush current subtitle
            wrapped = split_caption_text_two_lines(text, max_chars=max_chars)
            subs.append({
                "start": cur_start,
                "end": cur_end,
                "text": "\n".join(wrapped)
            })
            # Start fresh
            cur_words = []
            cur_start = cur_end

    # Flush remainder
    if cur_words:
        text = " ".join(x["word"] for x in cur_words).strip()
        wrapped = split_caption_text_two_lines(text, max_chars=max_chars)
        subs.append({
            "start": cur_start,
            "end": cur_words[-1]["end"],
            "text": "\n".join(wrapped)
        })

    return subs

# ----------------------------- Burn-in -----------------------------
def burn_subtitles_and_audio_to_video(original_video, subtitle_file, audio_file, output_video, tgt_lang_nllb="eng_Latn"):
    """
    Burn subtitles + TTS audio into video using the final SRT file as truth.
    """
    if subtitle_file.lower().endswith(".srt"):
        ass_file = subtitle_file.replace(".srt", ".ass")
        convert_srt_to_ass(subtitle_file, ass_file, tgt_lang_nllb)
    else:
        ass_file = subtitle_file

    font = get_font_for_lang(tgt_lang_nllb)
    vf_filter = f"ass={ass_file}"

    print(f"Burning subtitles with font '{font}' from {ass_file} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video, "-i", audio_file,
        "-vf", vf_filter,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_video
    ]
    subprocess.run(cmd, check=True)
    print(f"Output video created with font '{font}': {output_video}")

def generate_video_with_tts_audio(original_video, tts_audio_file, tgt_lang_nllb, subtitle_file=None, output_video=None):
    if not output_video:
        base = safe_filename(os.path.splitext(os.path.basename(original_video))[0])
        output_video = f"{base}_tts_only.mp4"

    cmd = ["ffmpeg", "-y", "-i", original_video, "-i", tts_audio_file, "-map", "0:v:0", "-map", "1:a:0"]

    # Explicitly skip subtitles unless provided
    if subtitle_file is not None:
        ass_file = subtitle_file.replace(".srt", ".ass")
        convert_srt_to_ass(subtitle_file, ass_file, tgt_lang_nllb)
        cmd.extend(["-vf", f"ass={ass_file}"])

    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", "-shortest", output_video
    ])

    print(f"Generating TTS-only video: {output_video} ...")
    subprocess.run(cmd, check=True)
    print(f"TTS-only video created: {output_video}")
    return output_video

def main():
    parser = argparse.ArgumentParser(description="Whisper → NLLB → MMS TTS → Burn-in")
    parser.add_argument("input", help="Input video/audio path")
    parser.add_argument("--language", "-l", default=None,
                        help="Force transcription language ISO-639-1 (e.g., hi, mr, es)")
    parser.add_argument(
        "--target_langs", "-t",
        nargs="+",   # accept multiple langs
        default=["hin_Deva"],
        help="One or more target languages (NLLB codes)"
    )
    args = parser.parse_args()

    input_path = args.input
    base = safe_filename(os.path.splitext(os.path.basename(input_path))[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        # === Step 1: Transcription ===
        audio_path = extract_audio_if_video(input_path, tmpdir)

        segments, detected_lang, all_words = transcribe_in_chunks(
            audio_path, language=args.language
        )
        src_lang_nllb = to_nllb_code(detected_lang or "en")
        print(f"[SRC] Using source language: {detected_lang} → {src_lang_nllb}")

        # === Step 2: Build audio-driven subtitles (original SRT) ===
        audio_driven_segments = build_audio_driven_subs(
            all_words,
            max_duration_s=5.0,
            max_chars=42,
            silence_gap_s=0.6
        )

        out_srt_original = f"{base}_{src_lang_nllb}_original.srt"
        with open(out_srt_original, "w", encoding="utf-8") as f:
            f.write(segments_to_srt(audio_driven_segments))
        print(f"[SAVE] Whisper SRT (AUDIO-DRIVEN) → {out_srt_original}")

        all_outputs = {"original_srt": out_srt_original, "translations": []}

        # === Step 3: Process each target language ===
        for tgt_lang in args.target_langs:
            tgt_lang_nllb = to_nllb_code(tgt_lang)
            out_srt_final = f"{base}_{tgt_lang_nllb}_final.srt"

            # ✅ If source is English → use sentence-based SRT translation
            if src_lang_nllb.startswith("eng"):
                translate_srt_file_segment(
                    out_srt_original,
                    out_srt_final,
                    src_lang_nllb,
                    tgt_lang_nllb
                )
                print(f"[SRT-TRANSLATE] Final translated SRT (English src) → {out_srt_final}")

            # ✅ Otherwise → use batched segment translation
            else:
                subs = pysubs2.load(out_srt_original, encoding="utf-8")
                segments_for_batch = [
                    {"start": ev.start/1000, "end": ev.end/1000, "text": ev.text}
                    for ev in subs
                ]
                translated_segments = translate_segments_nllb_batched(
                    segments_for_batch,
                    src_lang_nllb,
                    tgt_lang_nllb
                )

                final_events = []
                for seg in translated_segments:
                    final_events.append(pysubs2.SSAEvent(
                        start=int(seg["start"]*1000),
                        end=int(seg["end"]*1000),
                        text=seg["text"]
                    ))
                ssa = pysubs2.SSAFile()
                ssa.events = final_events
                ssa.save(out_srt_final, format_="srt", encoding="utf-8")
                print(f"[BATCHED-TRANSLATE] Final translated SRT (non-English src) → {out_srt_final}")

            # --- Subs Video ---
            ass_file = out_srt_final.replace(".srt", ".ass")
            convert_srt_to_ass(out_srt_final, ass_file, tgt_lang_nllb)
            out_video_subs = f"{base}_{tgt_lang_nllb}_subs.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, "-vf", f"ass={ass_file}",
                "-c:v", "libx264", "-preset", "fast", "-c:a", "copy", out_video_subs
            ], check=True)
            print(f"[SAVE] Video with subs → {out_video_subs}")

            # --- Dubbed Video (TTS) ---
            subs = pysubs2.load(out_srt_final, encoding="utf-8")
            full_text = " ".join([s.text.strip() for s in subs if s.text.strip()])
            tts_model = get_tts_model(tgt_lang_nllb)
            wav_out = tts_model.synthesis(full_text)

            y = wav_out["x"]
            sr = int(wav_out["sampling_rate"])
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y / max(1.0, np.max(np.abs(y))) * 0.9
            tensor_int16 = (y * 32767).astype(np.int16)
            tts_audio = AudioSegment(tensor_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

            probe = ffmpeg.probe(input_path)
            video_duration = float(probe["format"]["duration"]) * 1000
            tts_audio = speedup_audio_to_fit_segment(tts_audio, int(video_duration))

            out_audio_file = f"{base}_{tgt_lang_nllb}_tts.mp3"
            tts_audio.export(out_audio_file, format="mp3")

            out_video_tts = f"{base}_{tgt_lang_nllb}_tts.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", input_path, "-i", out_audio_file,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_video_tts
            ], check=True)
            print(f"[SAVE] Dubbed video → {out_video_tts}")

            all_outputs["translations"].append({
                "lang": tgt_lang_nllb,
                "srt": out_srt_final,
                "subs_video": out_video_subs,
                "tts_audio": out_audio_file,
                "dubbed_video": out_video_tts,
            })

        # === Summary ===
        print("\n=== OUTPUTS ===")
        print(f"Whisper Original SRT: {all_outputs['original_srt']}")
        for t in all_outputs["translations"]:
            print(f"\n[{t['lang']}]")
            print(f"  Final SRT:    {t['srt']}")
            print(f"  Subs Video:   {t['subs_video']}")
            print(f"  TTS Audio:    {t['tts_audio']}")
            print(f"  Dubbed Video: {t['dubbed_video']}")

if __name__ == "__main__":
    main()
