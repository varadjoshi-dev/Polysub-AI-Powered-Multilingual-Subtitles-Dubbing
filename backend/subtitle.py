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
<<<<<<< HEAD
# *** FIX: Removed 'from jobs import jobs' as it's no longer needed ***

# This file contains the core logic for video transcription, translation, and processing.
=======
from jobs import jobs
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f

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

<<<<<<< HEAD
    if len(lines) <= 2:
        return [ln.strip() for ln in lines]

=======
    # If already 1 or 2 lines -> good
    if len(lines) <= 2:
        return [ln.strip() for ln in lines]

    # More than 2 lines: merge into 2 lines trying to balance by char-count
    # Strategy: greedily pack words into first line until roughly half of total chars
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
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

<<<<<<< HEAD
=======

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

def dedupe_nearby_words(words, time_tol_s=0.05):
    """
    Remove word duplicates that arise from chunk overlap.
    Keeps the earliest occurrence when two words start almost at same time and are the same token.
    words: list of {"word","start","end"}
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w['start'], w['end']))
    out = [words_sorted[0]]
    for w in words_sorted[1:]:
        prev = out[-1]
        # if same normalized token and start times very close -> skip
        if abs(w['start'] - prev['start']) <= time_tol_s and _normalize_short_token(w['word']) == _normalize_short_token(prev['word']):
            # prefer the one with larger end (longer) to keep safer timing
            if w['end'] > prev['end']:
                out[-1] = w
            continue
        # also if this word entirely inside previous word interval and same token -> skip
        if w['start'] >= prev['start'] and w['end'] <= prev['end'] and _normalize_short_token(w['word']) == _normalize_short_token(prev['word']):
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

# ----------------------------- Brand Transliteration Map -----------------------------
BRAND_TRANSLITERATIONS = {
    "hin_Deva": {
        "Apple": "एप्पल",
        "Google": "गूगल",
        "Facebook": "फेसबुक",
        "Microsoft": "माइक्रोसॉफ्ट",
        "Amazon": "अमेज़न"
    },
    "mar_Deva": {
        "Apple": "एप्पल",
        "Google": "गूगल",
        "Facebook": "फेसबुक",
        "Microsoft": "मायक्रोसॉफ्ट",
        "Amazon": "अ\u095f\u0947\u092e\u093d\u094d\u091d\u094b\u0902"
    }
    # 🔹 Add more target languages as needed
}

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

def convert_srt_to_ass(srt_file, ass_file, tgt_lang_nllb):
    """
    Convert SRT → ASS with safe styling so no words are cut off.
    """
    with open(srt_file, "rb") as f:
        raw = f.read()

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Failed to decode subtitle file with common encodings")

    subs = pysubs2.SSAFile.from_string(text, encoding="utf-8", format_="srt")

    # Ensure style exists
    font = get_font_for_lang(tgt_lang_nllb)
    if "Default" not in subs.styles:
        subs.styles["Default"] = pysubs2.SSAStyle()

    # 🔹 Adjust style to avoid clipping
    subs.styles["Default"].fontname = font
    subs.styles["Default"].fontsize = 26    
    subs.styles["Default"].alignment = pysubs2.Alignment.BOTTOM_CENTER
    subs.styles["Default"].marginv = 50   # extra bottom margin
    subs.styles["Default"].marginl = 40   # left/right padding
    subs.styles["Default"].marginr = 40

    subs.save(ass_file, format_="ass")
    print(f"SRT converted to ASS with font '{font}' → {ass_file}")


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

def _preprocess_audio_for_whisper(src_path, target_sr=16000):
    """ Ensure file is mono 16k WAV and volume-normalized. Returns path to temp wav. Uses pydub (already in your imports). """
    audio = AudioSegment.from_file(src_path)
    # Force mono + sample rate
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    # Normalize loudness to avoid very quiet sections getting missed
    audio = effects.normalize(audio)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name
    tmp.close()
    audio.export(tmp_name, format="wav")
    return tmp_name

def _merge_and_dedupe_segments(segments, min_gap_s=0.05):
    """Merge Whisper segments conservatively. Avoids dropping words."""
    if not segments:
        return []
    segments_sorted = sorted(segments, key=lambda s: s['start'])
    merged = []
    cur = dict(segments_sorted[0])

    for s in segments_sorted[1:]:
        # Only merge if really overlapping or nearly touching
        if s['start'] <= cur['end'] + min_gap_s:
            cur['end'] = max(cur['end'], s['end'])
            # Append safely, avoid overwriting text
            if not cur['text'].strip().endswith(s['text'].strip()):
                cur['text'] = (cur['text'].strip() + " " + s['text'].strip()).strip()
        else:
            merged.append(cur)
            cur = dict(s)

    merged.append(cur)
    return merged


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
NLLB_MODEL_NAME="facebook/nllb-200-distilled-600M"
tokenizer=AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model=AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

def enforce_full_translation(text: str, src_lang: str, tgt_lang: str, translator) -> str:
    """
    Ensure every word is translated into target language.
    Detects leftover words (in Latin when tgt is non-Latin) and forces translation.
    """
    if not text.strip():
        return text

    # Tokenize words
    tokens = text.split()
    final_tokens = []

    for tok in tokens:
        # If target script is Devanagari and word still has A-Z → force translate
        if tgt_lang.endswith("_Deva") and re.search(r"[A-Za-z]", tok):
            try:
                translated_tok = translator(tok)[0]['translation_text']
                final_tokens.append(translated_tok.strip())
                continue
            except Exception:
                pass
        # If target is Arabic and token is Latin → force translate
        if tgt_lang.endswith("_Arab") and re.search(r"[A-Za-z]", tok):
            try:
                translated_tok = translator(tok)[0]['translation_text']
                final_tokens.append(translated_tok.strip())
                continue
            except Exception:
                pass
        # Otherwise keep
        final_tokens.append(tok)

    return " ".join(final_tokens)

def translate_segments_nllb(segments, src_lang_nllb, tgt_lang_nllb):
    translator = pipeline(
    "translation",
    model=nllb_model,
    tokenizer=tokenizer,
    src_lang=src_lang_nllb,
    tgt_lang=tgt_lang_nllb,
    max_length=2048   # 🔹 prevent truncation
)

    translated_segments = []
    print(f"Translating {len(segments)} segments {src_lang_nllb} → {tgt_lang_nllb}...")
    for seg in tqdm(segments):
        txt = seg['text'].strip()
        if not txt:
            translated_segments.append({'start': seg['start'], 'end': seg['end'], 'text': ""})
            continue
        translated_text = translator(txt)[0]['translation_text']
        translated_text = enforce_full_translation(translated_text, src_lang_nllb, tgt_lang_nllb, translator)

        # NOTE: deduplication intentionally NOT applied here to avoid double-processing —
        # it will be run once centrally in main() (unless --no-dedupe is passed).
        translated_segments.append({'start': seg['start'], 'end': seg['end'], 'text': translated_text})

    return translated_segments

def apply_brand_map(text: str, tgt_lang_nllb: str) -> str:
    brand_map = BRAND_TRANSLITERATIONS.get(tgt_lang_nllb, {})
    for eng_word, local_word in brand_map.items():
        # Replace standalone
        text = re.sub(rf"\b{eng_word}\b", local_word, text)
        # Replace "Apple + Devanagari" → keep Devanagari word
        text = re.sub(rf"{eng_word}\s*([ऀ-ॿ]+)", local_word + r" \1", text)
    return text


def dedupe_translations(text: str, tgt_lang_nllb: str = "eng_Latn") -> str:
    """
    Gentle deduplication:
    - Keeps valid repetitions (e.g., "very very good").
    - Removes stutter-like triples or mixed duplicates (e.g., "Google गूगल Google").
    - Applies brand map first.
    """
    if not text:
        return text

    text = apply_brand_map(text, tgt_lang_nllb)

    tokens = text.split()
    cleaned = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        # Check if next 2 tokens are the same → collapse
        if i + 1 < len(tokens) and tokens[i + 1].lower() == tok.lower():
            # Allow double, skip triple+
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

>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
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

<<<<<<< HEAD
def split_long_subtitles(input_srt, output_srt, max_chars=40):
    subs = pysubs2.load(input_srt, encoding="utf-8")
    new_subs = []
    for ev in subs:
        text = ev.text.strip()
        if not text:
            new_subs.append(ev)
            continue
        if len(text) <= max_chars:
            wrapped = split_caption_text_two_lines(text, max_chars=max_chars)
            ev.text = "\n".join(wrapped)
            new_subs.append(ev)
            continue
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
    if not tok:
        return ""
    core = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)
    return core.casefold()

def dedupe_nearby_words(words, time_tol_s=0.05):
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w['start'], w['end']))
    out = [words_sorted[0]]
    for w in words_sorted[1:]:
        prev = out[-1]
        if abs(w['start'] - prev['start']) <= time_tol_s and _normalize_short_token(w['word']) == _normalize_short_token(prev['word']):
            if w['end'] > prev['end']:
                out[-1] = w
            continue
        if w['start'] >= prev['start'] and w['end'] <= prev['end'] and _normalize_short_token(w['word']) == _normalize_short_token(prev['word']):
            continue
        out.append(w)
    return out
    
def adaptive_min_display(text, base_per_char=50, min_ms=800, max_ms=3000):
    est = len(text) * base_per_char
    return max(min_ms, min(est, max_ms))

def words_to_srt(words, max_chars=42, max_duration=5000):
    if not words:
        return ""
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
        if len(text) > max_chars or duration_ms > max_duration:
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
            last_w = cur_words[-1]
            cur_words = [last_w]
            cur_start = last_w['start']
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
    for i in range(1, len(subs)):
        if subs[i].start <= subs[i-1].start:
            subs[i].start = subs[i-1].end + datetime.timedelta(milliseconds=10)
        if subs[i].start >= subs[i].end:
            subs[i].end = subs[i].start + datetime.timedelta(milliseconds=50)
    return srt.compose(subs)

def safe_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def ensure_pkg(mod_name, pip_name=None):
    try:
        __import__(mod_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])

for pkg in ["transformers","sentencepiece","sacremoses","ttsmms","pydub","pysubs2","pysoundfile","inflect"]:
    ensure_pkg(pkg)

BRAND_TRANSLITERATIONS = {
    "hin_Deva": {"Apple": "एप्पल", "Google": "गूगल", "Facebook": "फेसबुक", "Microsoft": "माइक्रोसॉफ्ट", "Amazon": "अमेज़न"},
    "mar_Deva": {"Apple": "एप्पल", "Google": "गूगल", "Facebook": "फेसबुक", "Microsoft": "मायक्रोसॉफ्ट", "Amazon": "अ\u095f\u0947\u092e\u093d\u094d\u091d\u094b\u0902"}
}
p = inflect.engine()

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
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_paths = []
    start = 0
    while start < total_ms:
        end = min(start + chunk_length_ms, total_ms)
        chunk = audio[start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk.export(tmp.name, format="wav")
            chunk_paths.append((tmp.name, start))
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

LANG_FONTS = {
    "_Deva": "Noto Sans Devanagari", "_Beng": "Noto Sans Bengali", "_Gujr": "Noto Sans Gujarati",
    "_Guru": "Noto Sans Gurmukhi", "_Taml": "Noto Sans Tamil", "_Telu": "Noto Sans Telugu",
    "_Knda": "Noto Sans Kannada", "_Mlym": "Noto Sans Malayalam", "_Orya": "Noto Sans Oriya",
    "_Sinh": "Noto Sans Sinhala", "_Olck": "Noto Sans Ol Chiki", "_Laoo": "Noto Sans Lao",
    "_Hans": "Noto Sans SC", "_Hant": "Noto Sans TC", "_Jpan": "Noto Sans JP",
    "_Hang": "Noto Sans KR", "_Tibt": "Noto Sans Tibetan", "_Mymr": "Noto Sans Myanmar",
    "_Khmr": "Noto Sans Khmer", "_Thai": "Noto Sans Thai", "_Arab": "Noto Naskh Arabic",
    "_Hebr": "Noto Sans Hebrew", "_Syrc": "Noto Sans Syriac", "_Tfng": "Noto Sans Tifinagh",
    "_Cyrl": "Noto Sans", "_Glag": "Noto Sans Glagolitic", "_Latn": "Noto Sans",
    "_Grek": "Noto Sans Greek", "_Armn": "Noto Sans Armenian", "_Geor": "Noto Sans Georgian",
    "_Ethi": "Noto Sans Ethiopic", "_Cher": "Noto Sans Cherokee", "_Vaii": "Noto Sans Vai",
    "_Tale": "Noto Sans Tai Le", "_Talu": "Noto Sans New Tai Lue", "_Phag": "Noto Sans Phags Pa",
    "default": "Noto Sans"
}

def get_font_for_lang(tgt_lang_nllb: str) -> str:
    for suffix, font in LANG_FONTS.items():
        if tgt_lang_nllb.endswith(suffix):
            return font
    return LANG_FONTS["default"]

def convert_srt_to_ass(srt_file, ass_file, tgt_lang_nllb):
    try:
        subs = pysubs2.load(srt_file, encoding="utf-8")
    except Exception:
        with open(srt_file, "rb") as f:
            content = f.read().decode('utf-8', errors='ignore')
        subs = pysubs2.SSAFile.from_string(content)
        
    font = get_font_for_lang(tgt_lang_nllb)
    if "Default" not in subs.styles:
        subs.styles["Default"] = pysubs2.SSAStyle()
    subs.styles["Default"].fontname = font
    subs.styles["Default"].fontsize = 26
    subs.styles["Default"].alignment = pysubs2.Alignment.BOTTOM_CENTER
    subs.styles["Default"].marginv = 50
    subs.styles["Default"].marginl = 40
    subs.styles["Default"].marginr = 40
    subs.save(ass_file, format_="ass")
    print(f"SRT converted to ASS with font '{font}' → {ass_file}")

def speedup_audio_to_fit_segment(audio: AudioSegment, target_duration_ms: int, max_step=1.5) -> AudioSegment:
    current_duration = len(audio)
    if current_duration <= target_duration_ms:
        return audio + AudioSegment.silent(duration=(target_duration_ms - current_duration))
    speed_factor = current_duration / target_duration_ms
    if speed_factor <= 1.0:
        return audio
    step = 1.05
    while speed_factor > step and step < max_step:
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * step)})
        audio = audio.set_frame_rate(audio.frame_rate)
        speed_factor /= step
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
    return audio.set_frame_rate(audio.frame_rate)

def _preprocess_audio_for_whisper(src_path, target_sr=16000):
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    audio = effects.normalize(audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio.export(tmp.name, format="wav")
        return tmp.name

def _merge_and_dedupe_segments(segments, min_gap_s=0.05):
    if not segments:
        return []
    segments_sorted = sorted(segments, key=lambda s: s['start'])
    merged = [dict(segments_sorted[0])]
    for s in segments_sorted[1:]:
        cur = merged[-1]
        if s['start'] <= cur['end'] + min_gap_s:
            cur['end'] = max(cur['end'], s['end'])
            if not cur['text'].strip().endswith(s['text'].strip()):
                cur['text'] = (cur['text'].strip() + " " + s['text'].strip()).strip()
        else:
            merged.append(dict(s))
    return merged

ISO2_TO_NLLB = {
    "hi": "hin_Deva", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya",
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
}

def to_nllb_code(code: str) -> str:
    if "_" in code and len(code.split("_")[0])==3:
        return code
    return ISO2_TO_NLLB.get(code.lower(), "eng_Latn")

NLLB_TO_MMS_OVERRIDES={"zho":"cmn"}
def nllb_to_mms(nllb_code: str) -> str:
    return NLLB_TO_MMS_OVERRIDES.get(nllb_code.split("_")[0], nllb_code.split("_")[0])

def universal_normalize_text(text: str, lang_hint="eng") -> str:
    if not text:
        return text
    if not lang_hint.startswith("eng"):
        return re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+", " ", text).strip()

print("Loading NLLB model for translation...")
NLLB_MODEL_NAME="facebook/nllb-200-distilled-600M"
tokenizer=AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
nllb_model=AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

def enforce_full_translation(text: str, src_lang: str, tgt_lang: str, translator) -> str:
    if not text.strip():
        return text
    return text

def translate_segments_nllb(segments, src_lang_nllb, tgt_lang_nllb):
    translator = pipeline("translation", model=nllb_model, tokenizer=tokenizer, src_lang=src_lang_nllb, tgt_lang=tgt_lang_nllb, max_length=2048)
    translated_segments = []
    print(f"Translating {len(segments)} segments {src_lang_nllb} → {tgt_lang_nllb}...")
    for seg in tqdm(segments):
        txt = seg['text'].strip()
        if not txt:
            translated_segments.append({'start': seg['start'], 'end': seg['end'], 'text': ""})
            continue
        translated_text = translator(txt)[0]['translation_text']
        translated_segments.append({'start': seg['start'], 'end': seg['end'], 'text': translated_text})
    return translated_segments

def apply_brand_map(text: str, tgt_lang_nllb: str) -> str:
    brand_map = BRAND_TRANSLITERATIONS.get(tgt_lang_nllb, {})
    for eng_word, local_word in brand_map.items():
        text = re.sub(rf"\b{eng_word}\b", local_word, text)
    return text

def dedupe_translations(text: str, tgt_lang_nllb: str = "eng_Latn") -> str:
    if not text:
        return text
    text = apply_brand_map(text, tgt_lang_nllb)
    tokens = text.split()
    cleaned = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if i + 1 < len(tokens) and tokens[i + 1].lower() == tok.lower():
            if i + 2 < len(tokens) and tokens[i + 2].lower() == tok.lower():
                cleaned.append(tok)
                i += 3
                continue
        cleaned.append(tok)
        i += 1
    return " ".join(cleaned)

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

def tts_from_srt_global_fit_refined(srt_file, tgt_lang_nllb, video_path, output_audio, **kwargs):
=======
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

    # Apply brand transliteration only (do not dedupe here)
    text = apply_brand_map(text, tgt_lang_nllb)

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
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
    subs = pysubs2.load(srt_file, encoding="utf-8")
    full_text = " ".join([sub.text.strip() for sub in subs if sub.text.strip()])
    if not full_text:
        raise ValueError("No subtitle text to speak.")
    tts_model = get_tts_model(tgt_lang_nllb)
<<<<<<< HEAD
    if tgt_lang_nllb.startswith("eng"):
        full_text = universal_normalize_text(full_text)
    full_text = apply_brand_map(full_text, tgt_lang_nllb)
    wav_out = tts_model.synthesis(full_text)
    y, sr = wav_out["x"], int(wav_out["sampling_rate"])
    if y.ndim > 1: y = y.mean(axis=1)
    y = y / max(1.0, np.max(np.abs(y))) * 0.9
    tensor_int16 = (y * 32767).astype(np.int16)
    tts_audio = AudioSegment(tensor_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    tts_audio = AudioSegment.silent(200) + tts_audio + AudioSegment.silent(300)
    tts_audio = effects.normalize(tts_audio, headroom=3.0)
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe['format']['duration']) * 1000
    tts_audio = speedup_audio_to_fit_segment(tts_audio, int(video_duration))
=======
    # 🔹 Apply text normalization for English
    if tgt_lang_nllb.startswith("eng"):
        full_text = universal_normalize_text(full_text)

    # Apply brand transliteration only (do not dedupe here)
    full_text = apply_brand_map(full_text, tgt_lang_nllb)

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
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
    tts_audio.export(output_audio, format="mp3")
    print(f"Refined global-fit TTS file created: {output_audio}")
    return output_audio

<<<<<<< HEAD
def transcribe_in_chunks(audio_path, model_size='large', language=None, chunk_length_ms=60*1000, overlap_ms=1000):
    audio_path = _preprocess_audio_for_whisper(audio_path)
    chunk_info = split_audio(audio_path, chunk_length_ms, overlap_ms=overlap_ms)
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    segments, words_global, detected_lang = [], [], None
    for chunk_path, chunk_start_ms in tqdm(chunk_info, desc="Transcribing chunks"):
        result = model.transcribe(chunk_path, task="transcribe", word_timestamps=True, language=language)
        detected_lang = result.get("language", detected_lang)
        chunk_offset = chunk_start_ms / 1000.0
        for seg in result.get("segments", []):
            seg_start = float(seg['start']) + chunk_offset
            seg_end = float(seg['end']) + chunk_offset
            words = [{"word": w["word"].strip(), "start": float(w["start"]) + chunk_offset, "end": float(w["end"]) + chunk_offset} for w in seg.get("words", [])]
            new_seg = {"start": seg_start, "end": seg_end, "text": seg["text"].strip(), "words": words}
            segments.append(new_seg)
            words_global.extend(words)
    segments = _merge_and_dedupe_segments(segments)
    words_global = sorted(words_global, key=lambda w: (w['start'], w['end']))
    words_global = dedupe_nearby_words(words_global, time_tol_s=0.05)
    return segments, detected_lang, words_global

def refine_srt(input_srt, output_srt, max_line_length=42, lang_hint="eng"):
    subs = pysubs2.load(input_srt, encoding="utf-8")
    refined = []
    for i, ev in enumerate(subs):
        text = ev.text.strip()
        if lang_hint.startswith("eng"):
            text = re.sub(r"\b(uh|um|ah|er|hmm)\b", "", text, flags=re.IGNORECASE)
            text = universal_normalize_text(text, lang_hint=lang_hint)
        text = re.sub(r"\s+", " ", text).strip()
        if lang_hint.startswith("eng") and text:
            if not text.endswith((".", "?", "!")): text += "."
            text = text[0].upper() + text[1:]
        start, end = ev.start, ev.end
        if i > 0 and start < refined[-1].end: start = refined[-1].end + 10
        min_disp = adaptive_min_display(text)
        if end - start < min_disp: end = start + min_disp
        ev.text = "\n".join(split_caption_text_two_lines(text, max_chars=max_line_length))
        ev.start, ev.end = start, end
        refined.append(ev)
    for i in range(1, len(refined)):
        if refined[i].start < refined[i-1].end:
            refined[i].start = refined[i-1].end + 10
=======
# ----------------------------- Whisper -----------------------------
def transcribe_in_chunks(
    audio_path,
    model_size='large',
    language=None,
    chunk_length_ms=60 * 1000,   # larger chunks preserve context
    overlap_ms=1000              # overlap ensures continuity
):
    """
    High-accuracy chunked Whisper transcription:
    - Preprocess audio (mono 16k normalized WAV)
    - Split into chunks with overlap to avoid boundary word loss
    - Use Whisper with word-level timestamps
    - Deduplicate overlapping words safely
    - Merge segments conservatively for sync
    Returns:
        segments: list of {'start','end','text','words'}
        detected_lang: ISO-639-1 code
        words_global: list of {'word','start','end'} (deduped)
    """
    # --- Step 1: Preprocess audio ---
    audio_path = _preprocess_audio_for_whisper(audio_path)

    print(f"Splitting audio into {chunk_length_ms // 1000}s chunks with {overlap_ms}ms overlap...")
    chunk_info = split_audio(audio_path, chunk_length_ms, overlap_ms=overlap_ms)

    # --- Step 2: Load Whisper ---
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    segments = []
    words_global = []
    detected_lang = None

    # --- Step 3: Process each chunk ---
    for idx, (chunk_path, chunk_start_ms) in enumerate(tqdm(chunk_info, desc="Transcribing chunks")):
        options = {"task": "transcribe", "word_timestamps": True}
        if language:
            options['language'] = language

        result = model.transcribe(chunk_path, **options)
        detected_lang = result.get("language", detected_lang)

        chunk_offset = chunk_start_ms / 1000.0  # ms → seconds
        prev_end = segments[-1]["end"] if segments else None

        for seg in result.get("segments", []):
            seg_start = float(seg['start']) + chunk_offset
            seg_end = float(seg['end']) + chunk_offset

            if prev_end and seg_start < prev_end and "words" in seg:
                # Handle overlap at word level
                words = [
                    {
                        "word": w["word"].strip(),
                        "start": float(w["start"]) + chunk_offset,
                        "end": float(w["end"]) + chunk_offset
                    }
                    for w in seg["words"]
                    if float(w["end"]) + chunk_offset > prev_end
                ]
                if not words:
                    continue
                seg_start = words[0]["start"]
                seg_end = words[-1]["end"]
                seg_text = " ".join(w["word"] for w in words)
                new_seg = {"start": seg_start, "end": seg_end, "text": seg_text, "words": words}
                words_global.extend(words)
            else:
                new_seg = {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg["text"].strip(),
                    "words": [
                        {
                            "word": w["word"].strip(),
                            "start": float(w["start"]) + chunk_offset,
                            "end": float(w["end"]) + chunk_offset
                        }
                        for w in seg.get("words", [])
                    ]
                }
                words_global.extend(new_seg["words"])

            segments.append(new_seg)
            prev_end = seg_end

    # --- Step 4: Merge and dedupe ---
    segments = _merge_and_dedupe_segments(segments)

    # Deduplicate word timeline from overlaps
    words_global = sorted(words_global, key=lambda w: (w['start'], w['end']))
    words_global = dedupe_nearby_words(words_global, time_tol_s=0.05)

    # Normalize values (defensive)
    for w in words_global:
        w['start'] = float(w['start'])
        w['end'] = float(w['end'])
        w['word'] = w['word'].strip()

    return segments, detected_lang, words_global

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

def refine_srt(input_srt, output_srt, max_line_length=42, min_display_time=800, lang_hint="eng"):
    """
    Ensure subtitles are safe for ffmpeg renderer:
    - Minimum display duration
    - Forced 1-2 line wrapping (so ffmpeg never skips words)
    - No missing words
    """
    subs = pysubs2.load(input_srt, encoding="utf-8")
    refined = []

    for i, ev in enumerate(subs):
        text = ev.text.strip()

        # Cleanup language-specific
        if lang_hint.startswith("eng"):
            text = re.sub(r"\b(uh|um|ah|er|hmm)\b", "", text, flags=re.IGNORECASE)
            text = universal_normalize_text(text, lang_hint=lang_hint)

        text = re.sub(r"\s+", " ", text).strip()

        # Guarantee punctuation and capitalization
        if lang_hint.startswith("eng") and text:
            if not text.endswith((".", "?", "!")):
                text += "."
            text = text[0].upper() + text[1:]

        # Timing fixes
        start, end = ev.start, ev.end
        if i > 0 and start < refined[-1].end:
            start = refined[-1].end + 10

        # Enforce minimum display duration
        min_disp = adaptive_min_display(text)
        if end - start < min_disp:
            end = start + min_disp

        # Force 1-2 line wrapping (so ffmpeg never skips words)
        wrapped_lines = split_caption_text_two_lines(text, max_chars=max_line_length)
        text = "\n".join(wrapped_lines)

        ev.text = text
        ev.start, ev.end = start, end
        refined.append(ev)

    # Ensure strictly increasing timings
    for i in range(1, len(refined)):
        if refined[i].start < refined[i-1].end:
            refined[i].start = refined[i-1].end + 10

>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
    ssa_file = pysubs2.SSAFile()
    ssa_file.events = refined
    ssa_file.save(output_srt, encoding="utf-8", format_="srt")
    print(f"[SAFE] Refined subtitles saved → {output_srt}")

def _normalize_token_for_compare(tok: str) -> str:
<<<<<<< HEAD
    return regex.sub(r'^\p{P}+|\p{P}+$', '', tok).casefold()

def collapse_repeated_runs_in_text(text: str) -> str:
    if not text: return text
    toks = text.split()
    out, prev_norm = [], None
    for t in toks:
        norm = _normalize_token_for_compare(t)
        if norm == prev_norm: continue
=======
    """Return a normalized token for equality checks (strip punctuation, casefold)."""
    # remove leading/trailing punctuation (unicode-aware) and casefold
    core = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)
    return core.casefold()

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
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
        out.append(t)
        prev_norm = norm
    return " ".join(out)

def _remove_boundary_duplicates(segments):
<<<<<<< HEAD
    if not segments: return segments
    segments_sorted = sorted(segments, key=lambda s: s['start'])
    for i in range(len(segments_sorted) - 1):
        a_tokens, b_tokens = segments_sorted[i]['text'].split(), segments_sorted[i+1]['text'].split()
        if not a_tokens or not b_tokens: continue
        max_k = min(3, len(a_tokens), len(b_tokens))
        for k in range(max_k, 0, -1):
            if [_normalize_token_for_compare(x) for x in a_tokens[-k:]] == [_normalize_token_for_compare(x) for x in b_tokens[:k]]:
                segments_sorted[i+1]['text'] = " ".join(b_tokens[k:]).strip()
                break
    return segments_sorted

def tidy_translated_segments(segments):
    for seg in segments:
        seg['text'] = collapse_repeated_runs_in_text(seg.get('text','')).strip()
    segments = _remove_boundary_duplicates(segments)
=======
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
      - Each word or small group of words is translated into target language
      - Timing from Whisper preserved
    """
    subs = []
    cur_words = []
    start = None

    for w in words:
        if not cur_words:
            start = w['start']
        cur_words.append(w)

        text_en = " ".join(x['word'].strip() for x in cur_words).strip()
        duration = (w['end'] - start) * 1000

        if (len(text_en) > max_chars) or (duration > max_duration):
            # Close current subtitle before the last word
            end = cur_words[-2]['end']
            text_en = " ".join(x['word'].strip() for x in cur_words[:-1])

            # 🔹 Translate chunk
            translated = translator(text_en)[0]['translation_text']
            translated = enforce_full_translation(translated, src_lang_nllb, tgt_lang_nllb, translator)

            subs.append(srt.Subtitle(
                index=len(subs) + 1,
                start=datetime.timedelta(seconds=start),
                end=datetime.timedelta(seconds=end),
                content="\n".join(split_caption_text_two_lines(translated, max_chars))
            ))

            # Start new subtitle with last word
            cur_words = [cur_words[-1]]
            start = cur_words[0]['start']

    if cur_words:
        text_en = " ".join(x['word'].strip() for x in cur_words)
        translated = translator(text_en)[0]['translation_text']
        translated = enforce_full_translation(translated, src_lang_nllb, tgt_lang_nllb, translator)

        subs.append(srt.Subtitle(
            index=len(subs) + 1,
            start=datetime.timedelta(seconds=start),
            end=datetime.timedelta(seconds=cur_words[-1]['end']),
            content="\n".join(split_caption_text_two_lines(translated, max_chars))
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
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
    for seg in segments:
        seg['text'] = seg['text'].strip()
    return segments

<<<<<<< HEAD
# *** FIX: Updated function signature to accept 'jobs' dictionary ***
def process(jobs, input_path, **kwargs):
    job_id = kwargs.get("job_id")
    enableTts = kwargs.get("enableTts", False)
    target_lang = kwargs.get("target_lang", "hin_Deva")
    model_size = kwargs.get("model", "small") # Default to small model
    
    def update(progress, status, message):
        if job_id and job_id in jobs:
            jobs[job_id]["progress"] = progress
            jobs[job_id]["status"] = status
            jobs[job_id]["logs"].append(message)
            print(f"[{job_id}] {status} ({progress}%): {message}")

    try:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")

        base = os.path.splitext(os.path.basename(input_path))[0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            update(5, "processing", "Extracting audio from video...")
            audio_path = extract_audio_if_video(input_path, tmpdir)

            update(10, "processing", "Transcribing audio to text...")
            segments, detected_lang_iso2, _ = transcribe_in_chunks(
                audio_path, model_size=model_size
            )
            if not segments:
                update(100, "failed", "Transcription failed: No text detected.")
                return

            src_lang_nllb = to_nllb_code(detected_lang_iso2 or "en")
            update(30, "processing", f"Detected language: {src_lang_nllb}")

            targets_raw = [x.strip() for x in target_lang.split(",")]
            target_langs_nllb = [to_nllb_code(t) for t in targets_raw]

            for i, tgt_lang_nllb in enumerate(target_langs_nllb):
                lang_progress_start = 30 + (i * (70 / len(target_langs_nllb)))
                
                if tgt_lang_nllb == src_lang_nllb:
                    update(lang_progress_start + 5, "processing", f"Skipping translation for {tgt_lang_nllb} (same as source).")
                    continue

                update(lang_progress_start + 10, "processing", f"Translating to {tgt_lang_nllb}...")
                translated_segments = translate_segments_nllb(segments, src_lang_nllb, tgt_lang_nllb)
                translated_segments = tidy_translated_segments(translated_segments)
                
                update(lang_progress_start + 25, "processing", "Generating subtitle file...")
                out_srt_final = f"{base}_{tgt_lang_nllb}_final.srt"
                with open(out_srt_final, "w", encoding="utf-8") as f:
                    f.write(segments_to_srt(translated_segments))
                refine_srt(out_srt_final, out_srt_final, lang_hint=tgt_lang_nllb)
                
                ass_file_final = out_srt_final.replace(".srt", ".ass")
                convert_srt_to_ass(out_srt_final, ass_file_final, tgt_lang_nllb)
                
                update(lang_progress_start + 40, "processing", "Creating subtitled video...")
                out_video_subs = f"{base}_{tgt_lang_nllb}_subs.mp4"
                cmd_subs = ["ffmpeg", "-y", "-i", input_path, "-vf", f"ass={ass_file_final}", "-c:v", "libx264", "-preset", "fast", "-c:a", "copy", out_video_subs]
                subprocess.run(cmd_subs, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # *** FIX: Check for enableTts before generating TTS video ***
                if enableTts:
                    update(lang_progress_start + 50, "processing", "Generating dubbed audio...")
                    tts_audio_file = f"{base}_{tgt_lang_nllb}_tts.mp3"
                    tts_from_srt_global_fit_refined(out_srt_final, tgt_lang_nllb, input_path, tts_audio_file)
                    
                    update(lang_progress_start + 60, "processing", "Creating dubbed video...")
                    out_video_tts = f"{base}_{tgt_lang_nllb}_tts.mp4"
                    cmd_tts = ["ffmpeg", "-y", "-i", input_path, "-i", tts_audio_file, "-map", "0:v:0", "-map", "1:a:0", "-vf", f"ass={ass_file_final}", "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-b:a", "192k", "-shortest", out_video_tts]
                    subprocess.run(cmd_tts, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        update(100, "done", "Processing finished successfully!")

    except Exception as e:
        print(f"Error during processing: {e}")
        update(100, "failed", f"An error occurred: {str(e)}")

=======

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

def process(input_path, **kwargs):

    job_id = kwargs.get("job_id")
    enableTts = kwargs.get("enableTts", False)
    enableRealtime = kwargs.get("enableRealtime", False)
    generateSrt = kwargs.get("generateSrt", True)
    target_lang = kwargs.get("target_lang", "hin_Deva")
    model="large"
    language=None
    speed=1.0
    no_dedupe=False
    debug=False

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    base = safe_filename(os.path.splitext(os.path.basename(input_path))[0])

    # init job
    if job_id:
        jobs[job_id]["status"] = "queued"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["logs"] = []

    def update(progress, status, message):
        """Update job status & logs."""
        if job_id:
            jobs[job_id]["progress"] = progress
            jobs[job_id]["status"] = status
            jobs[job_id]["logs"].append(message)
        print(f"[{job_id}] {message}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = extract_audio_if_video(input_path, tmpdir)

        # === Step 1: Transcription ===
        update(10, "upload", "Running Whisper transcription...")
        segments, detected_lang_iso2, words_global = transcribe_in_chunks(
            audio_path, model_size=model, language=language, chunk_length_ms=30 * 1000
        )
        if not segments:
            update(100, "failed", "No transcription returned. Job failed.")
            jobs[job_id]["status"] = "failed"
            return

        src_lang_nllb = to_nllb_code(detected_lang_iso2 or "en")
        update(20, "upload",f"Detected language: {detected_lang_iso2} → {src_lang_nllb}")

        targets_raw = [x.strip() for x in target_lang.split(",")]
        target_langs_nllb = [to_nllb_code(t) for t in targets_raw]

        for tgt_lang_nllb in target_langs_nllb:
            if tgt_lang_nllb == src_lang_nllb:
                update(25, "asr",f"Skipping {tgt_lang_nllb}, same as source.")
                continue

            # === Step 2: Translation ===
            update(50, "asr",f"Translating → {tgt_lang_nllb}")
            try:
                translated_segments = translate_segments_nllb(segments, src_lang_nllb, tgt_lang_nllb)
            except Exception as e:
                update(100, "failed",f"[WARN] Translation failed for {tgt_lang_nllb}: {e}")
                jobs[job_id]["status"] = "failed"
                return

            # === Deduplication ===
            if not no_dedupe:
                for seg in translated_segments:
                    seg["text"] = dedupe_translations(seg["text"], tgt_lang_nllb)

            translated_segments = tidy_translated_segments(translated_segments)

            # === Step 1: RAW translated SRT (segment-level) ===
            update(60, "asr", "Generating SRT files...")
            out_srt_raw = f"{base}_{tgt_lang_nllb}_raw.srt"
            with open(out_srt_raw, "w", encoding="utf-8") as f:
                f.write(segments_to_srt(translated_segments))
            print(f"RAW SRT file created: {out_srt_raw}")

            # 🔹 Fix long lines
            split_long_subtitles(out_srt_raw, out_srt_raw, max_chars=40)

            # === Step 2: Final refined SRT ===
            out_srt_final = f"{base}_{tgt_lang_nllb}_final.srt"
            shutil.copy(out_srt_raw, out_srt_final)
            refine_srt(out_srt_final, out_srt_final, lang_hint=tgt_lang_nllb)
            print(f"Final refined SRT created: {out_srt_final}")
            update(75, "translate", "refined SRT created...")

            # Convert FINAL SRT → ASS
            ass_file_final = out_srt_final.replace(".srt", ".ass")
            convert_srt_to_ass(out_srt_final, ass_file_final, tgt_lang_nllb)

            # === Step 3: Burn video with FINAL refined subs ===

#             if()

            out_video_subs = f"{base}_{tgt_lang_nllb}_subs.mp4"
            cmd_subs = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", f"ass={ass_file_final}",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "copy",
                out_video_subs
            ]
            subprocess.run(cmd_subs, check=True)
            update(80, "translate","Video with refined subtitles created...")
            print(f"Video with refined subtitles created: {out_video_subs}")

            # === Step 4: Generate TTS audio from refined subs ===
            update(80, "subtitle","Burning subtitles into video...")
            subs = pysubs2.load(out_srt_final, encoding="utf-8")
            full_text = " ".join([sub.text.strip() for sub in subs if sub.text.strip()])
            full_text = collapse_repeated_runs_in_text(full_text)

            tts_model = get_tts_model(tgt_lang_nllb)
            wav_out = tts_model.synthesis(full_text)
            y = wav_out["x"]
            sr = int(wav_out["sampling_rate"])
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y / max(1.0, np.max(np.abs(y))) * 0.9
            tensor_int16 = (y * 32767).astype(np.int16)
            tts_audio = AudioSegment(tensor_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
            tts_audio = AudioSegment.silent(duration=200) + tts_audio + AudioSegment.silent(duration=300)
            if len(tts_audio) > 160:
                tts_audio = tts_audio.fade_in(80).fade_out(80)
            tts_audio = effects.normalize(tts_audio, headroom=3.0)

            probe = ffmpeg.probe(input_path)
            video_duration = float(probe['format']['duration']) * 1000
            tts_audio = speedup_audio_to_fit_segment(tts_audio, int(video_duration))

            tts_audio_io = io.BytesIO()
            tts_audio.export(tts_audio_io, format="mp3")
            tts_audio_io.seek(0)

            out_video_tts = f"{base}_{tgt_lang_nllb}_tts.mp4"
            cmd_tts = [
                "ffmpeg", "-y", "-i", input_path, "-i", "pipe:0",
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", f"ass={ass_file_final}",   # ✅ always use final refined subs
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_video_tts
            ]
            process = subprocess.Popen(cmd_tts, stdin=subprocess.PIPE)
            process.communicate(tts_audio_io.read())
            print(f"TTS video created (refined subs burned in): {out_video_tts}")

            # === Summary ===
            print(f"[DONE] {tgt_lang_nllb}:")
            print(f" → Final refined SRT: {out_srt_final}")
            print(f" → Video with refined subs: {out_video_subs}")
            print(f" → TTS Video with refined subs: {out_video_tts}")

        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "done"
        update(100, "✅ Job finished successfully")


def main():
    parser = argparse.ArgumentParser(description="Whisper → NLLB → MMS TTS → Burn-in")
    parser.add_argument("input", help="Input video/audio path")
    parser.add_argument("--model", "-m", default="large", help="Whisper model size")
    parser.add_argument("--language", "-l", default=None, help="Force transcription language ISO-639-1")
    parser.add_argument("--target_lang", "-t", default="hin_Deva", help="Comma-separated target languages")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Base TTS speed multiplier")
    parser.add_argument("--no-dedupe", dest="no_dedupe", action="store_true", help="Disable deduplication step")
    parser.add_argument("--debug", action="store_true", help="Print before/after dedupe for first few segments")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input not found: {input_path}")

    base = safe_filename(os.path.splitext(os.path.basename(input_path))[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = extract_audio_if_video(input_path, tmpdir)

        # ✅ transcription
        segments, detected_lang_iso2, words_global = transcribe_in_chunks(
            audio_path, model_size=args.model, language=args.language, chunk_length_ms=30 * 1000
        )
        if not segments:
            print("No transcription returned.")
            return

        src_lang_nllb = to_nllb_code(detected_lang_iso2 or "en")
        print(f"Detected language: {detected_lang_iso2} → {src_lang_nllb}")

        targets_raw = [x.strip() for x in args.target_lang.split(",")]
        target_langs_nllb = [to_nllb_code(t) for t in targets_raw]

        for tgt_lang_nllb in target_langs_nllb:
            if tgt_lang_nllb == src_lang_nllb:
                print(f"Skipping {tgt_lang_nllb}, same as source.")
                continue

            # === Translation ===
            try:
                translated_segments = translate_segments_nllb(segments, src_lang_nllb, tgt_lang_nllb)
            except Exception as e:
                print(f"[WARN] Translation failed for {tgt_lang_nllb}: {e}")
                continue

            # === Deduplication ===
            if not args.no_dedupe:
                for seg in translated_segments:
                    seg["text"] = dedupe_translations(seg["text"], tgt_lang_nllb)

            translated_segments = tidy_translated_segments(translated_segments)

            # === Step 1: RAW translated SRT (segment-level) ===
            out_srt_raw = f"{base}_{tgt_lang_nllb}_raw.srt"
            with open(out_srt_raw, "w", encoding="utf-8") as f:
                f.write(segments_to_srt(translated_segments))
            print(f"RAW SRT file created: {out_srt_raw}")

            # 🔹 Fix long lines
            split_long_subtitles(out_srt_raw, out_srt_raw, max_chars=40)

            # === Step 2: Final refined SRT ===
            out_srt_final = f"{base}_{tgt_lang_nllb}_final.srt"
            shutil.copy(out_srt_raw, out_srt_final)
            refine_srt(out_srt_final, out_srt_final, lang_hint=tgt_lang_nllb)
            print(f"Final refined SRT created: {out_srt_final}")

            # Convert FINAL SRT → ASS
            ass_file_final = out_srt_final.replace(".srt", ".ass")
            convert_srt_to_ass(out_srt_final, ass_file_final, tgt_lang_nllb)

            # === Step 3: Burn video with FINAL refined subs ===
            out_video_subs = f"{base}_{tgt_lang_nllb}_subs.mp4"
            cmd_subs = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", f"ass={ass_file_final}",
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "copy",
                out_video_subs
            ]
            subprocess.run(cmd_subs, check=True)
            print(f"Video with refined subtitles created: {out_video_subs}")

            # === Step 4: Generate TTS audio from refined subs ===
            subs = pysubs2.load(out_srt_final, encoding="utf-8")
            full_text = " ".join([sub.text.strip() for sub in subs if sub.text.strip()])
            full_text = collapse_repeated_runs_in_text(full_text)

            tts_model = get_tts_model(tgt_lang_nllb)
            wav_out = tts_model.synthesis(full_text)
            y = wav_out["x"]
            sr = int(wav_out["sampling_rate"])
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y / max(1.0, np.max(np.abs(y))) * 0.9
            tensor_int16 = (y * 32767).astype(np.int16)
            tts_audio = AudioSegment(tensor_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
            tts_audio = AudioSegment.silent(duration=200) + tts_audio + AudioSegment.silent(duration=300)
            if len(tts_audio) > 160:
                tts_audio = tts_audio.fade_in(80).fade_out(80)
            tts_audio = effects.normalize(tts_audio, headroom=3.0)

            probe = ffmpeg.probe(input_path)
            video_duration = float(probe['format']['duration']) * 1000
            tts_audio = speedup_audio_to_fit_segment(tts_audio, int(video_duration))

            tts_audio_io = io.BytesIO()
            tts_audio.export(tts_audio_io, format="mp3")
            tts_audio_io.seek(0)

            out_video_tts = f"{base}_{tgt_lang_nllb}_tts.mp4"
            cmd_tts = [
                "ffmpeg", "-y", "-i", input_path, "-i", "pipe:0",
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", f"ass={ass_file_final}",   # ✅ always use final refined subs
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_video_tts
            ]
            process = subprocess.Popen(cmd_tts, stdin=subprocess.PIPE)
            process.communicate(tts_audio_io.read())
            print(f"TTS video created (refined subs burned in): {out_video_tts}")

            # === Summary ===
            print(f"[DONE] {tgt_lang_nllb}:")
            print(f" → Final refined SRT: {out_srt_final}")
            print(f" → Video with refined subs: {out_video_subs}")
            print(f" → TTS Video with refined subs: {out_video_tts}")


if __name__ == "__main__":
    main()
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
