# ✨ PolySub – AI-Powered Multilingual Subtitles & Dubbing

**PolySub – AI-Powered Multilingual Subtitles & Dubbing**  
Translate, subtitle, and dub your videos in 200+ languages instantly.  

🔗 **Live Demo:** [polysub.netlify.app](https://polysub.netlify.app)  
## 🚀 How It Works
 - 📤 Upload a video  
 - 🎙 Speech Recognition (OpenAI Whisper)  
 - 🌐 Translation (Meta’s NLLB)  
 - 📝 Subtitle Generation  . 🗣 Dubbing (Microsoft Neural TTS via edge-tts)  
- 📥 Download fully processed video  

## ✨ Features
- 🎥 Upload videos in `.mp4`, `.mov`, `.avi`, `.mkv` formats  
- 🎙 Accurate speech-to-text transcription (**OpenAI Whisper**)  
- 🌐 Translation into **200+ languages** (Meta’s **NLLB**)  
- 🗣 Natural dubbing with **edge-tts** (Microsoft Neural TTS)  
- 📄 Export subtitles in `.srt` formats  
- 🔊 Optional real-time subtitles (for streaming/meetings)  

## 🛠 Tech Stack
- **Frontend:** Next.js, React, TailwindCSS  
- **AI Models:** OpenAI Whisper (ASR), Meta NLLB (Translation), Microsoft Neural TTS  
- **Media Processing:** FFmpeg  
- **Deployment:** Netlify  

## 📦 Installation (Local Setup)
```bash
# Clone repository
git clone https://github.com/varadjoshi08/Polysub-AI-Powered-Multilingual-Subtitles-Dubbing.git
```

```bash
# Go into project folder
cd Polysub-AI-Powered-Multilingual-Subtitles-Dubbing/frontend
```
```bash
# Install dependencies
pnpm install
```

```bash
# Start development server
pnpm dev
```

```bash
# Build for production
pnpm build
```
```bash
# Start production server
pnpm start
```


    
## 📊 Evaluation
- ✅ **Word Error Rate (WER):** transcription accuracy  
- ✅ **BLEU Score:** translation quality  
- ⏱ **Latency:** processing & response speed  
- 📝 **Subtitle Sync Score:** alignment accuracy between audio & subtitles
  
## Acknowledgements
 - [🧠 OpenAI Whisper – Speech recognition](https://github.com/openai/whisper)
 - [🌐 Meta NLLB – Multilingual translation](https://ai.meta.com/research/no-language-left-behind/)
 - [🔊 Microsoft edge-tts – Natural voice synthesis](https://github.com/rany2/edge-tts)
 - [🎬 FFmpeg – Media processing](https://ffmpeg.org/)

