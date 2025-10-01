# Powered-Speech-Annotation-and-Multilingual-Subtitle-Generation-System
# Powered Speech Annotation & Multilingual Subtitle + Dubbing System  

An **AI-powered end-to-end system** that:  
- Transcribes speech from video/audio using **OpenAI Whisper**  
- Translates subtitles into **200+ languages** using **Meta’s NLLB-200**  
- Generates **time-synced subtitles** (`.srt` / `.ass`)  
- Produces **refined TTS audio dubbing** in the target language using **MMS TTS**  
- Burns subtitles + dubbed audio into the video for a **final multilingual output**  

This enables **seamless multilingual communication and accessibility** for lectures, meetings, tutorials, and video content.  

---

## Features  
- Speech-to-text transcription (Whisper `large-v3`)  
- Translation into 200+ languages (NLLB-200)  
- Subtitle generation with intelligent line wrapping & timing fixes  
- MMS-based TTS dubbing (`.mp3`) with adaptive timing, fades, and normalization  
- Burn-in subtitles & dubbed audio into video using `ffmpeg`  
- Works with both **video and audio files**  

---

# Basic usage [English to Hindi]
- python subtitle.py input_video.mp4 --target_lang hi

# Multiple translations [English to Hindi, Spanish, French]
- python subtitle.py input_video.mp4 --target_langs hi es fr

# Force a specific Whisper model size [faster ones: base, small, medium; most accurate: large]
- python subtitle.py input_video.mp4 --model medium

# Force transcription language [if Whisper is unsure]
- python subtitle.py input_video.mp4 --language fr

---

# Output Files
- For each target language (<lang> = NLLB code, e.g., hin_Deva):
- <base>_eng_Latn_original.srt → Whisper English subtitles
- <base>_<lang>_final.srt → Final translated subtitles
- <base>_<lang>_subs.mp4 → Video with subtitles only
- <base>_<lang>_tts.mp4 → Final dubbed video with target audio

---

# Supported Languages

- Whisper auto-detects 90+ input languages
- NLLB supports 150+ target languages
- MMS provides TTS voices for 150+ of languages

---

# Note 

- The models are large (Whisper large-v3 ~3GB, NLLB-200 ~12GB).
- First run will download models automatically.
