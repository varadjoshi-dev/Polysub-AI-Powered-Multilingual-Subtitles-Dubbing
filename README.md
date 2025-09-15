# 🌍 PolySub  

**PolySub – AI-Powered Multilingual Subtitles & Dubbing**  
Translate, subtitle, and dub your videos in 200+ languages instantly.  

🔗 **Live Demo:** [polysub.netlify.app](https://polysub.netlify.app)  

---

## 🚀 How It Works
1. 📤 Upload a video  
2. 🎙 Speech Recognition (OpenAI Whisper)  
3. 🌐 Translation (Meta’s NLLB)  
4. 📝 Subtitle Generation  
5. 🗣 Dubbing (Microsoft Neural TTS via edge-tts)  
6. 📥 Download fully processed video  

---

## ✨ Features
- 🎥 Upload videos in `.mp4`, `.mov`, `.avi`, `.mkv` formats  
- 🎙 Accurate speech-to-text transcription (**OpenAI Whisper**)  
- 🌐 Translation into **200+ languages** (Meta’s **NLLB**)  
- 🗣 Natural dubbing with **edge-tts** (Microsoft Neural TTS)  
- 📄 Export subtitles in `.srt`, `.vtt`, `.txt` formats  
- 🔊 Optional real-time subtitles (for streaming/meetings)  
- ♿ Accessibility support for Deaf/Hard-of-Hearing users  

---

## 🛠 Tech Stack
- **Frontend:** Next.js, React, TailwindCSS  
- **AI Models:** OpenAI Whisper (ASR), Meta NLLB (Translation), Microsoft Neural TTS  
- **Media Processing:** FFmpeg  
- **Deployment:** Netlify  

---

## 📦 Installation (Local Setup)

```bash
# Clone repository
git clone https://github.com/varadjoshi08/Polysub-AI-Powered-Multilingual-Subtitles-Dubbing.git

# Go into project folder
cd Polysub-AI-Powered-Multilingual-Subtitles-Dubbing/frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start**


📊 Evaluation

✅ Word Error Rate (WER) for transcription accuracy

✅ BLEU Score for translation quality

✅ MOS (Mean Opinion Score) for dubbing naturalness

🗺 Roadmap

🧑‍🤝‍🧑 Add speaker diarization (multi-speaker support)

📡 Add support for live streaming input

☁️ Cloud storage integration (Google Drive, Dropbox)

🗣 More dubbing voices (custom voice cloning)

🙌 Acknowledgements

🧠 OpenAI Whisper – Speech recognition

🌐 Meta NLLB – Multilingual translation

🔊 Microsoft edge-tts – Natural voice synthesis

🎬 FFmpeg – Media processing

📜 License

This project is licensed under the MIT License.

👨‍💻 Author

Built with ❤️ by Varad Joshi
