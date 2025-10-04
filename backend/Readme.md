AI-Powered Multilingual Video Dubbing & Subtitling Backend
This powerful backend script automates the entire process of creating multilingual subtitles and dubbed audio for any video or audio file. It uses a state-of-the-art AI pipeline to deliver high-quality, synchronized results with minimal manual effort.

✨ Features
Automatic Transcription: Utilizes OpenAI's Whisper to generate highly accurate, word-level timestamps from the source audio.

AI-Powered Translation: Leverages Meta's NLLB (No Language Left Behind) model to translate text into hundreds of languages.

Text-to-Speech Dubbing: Generates natural-sounding dubbed audio tracks in the target language using Meta's MMS TTS models.

Intelligent Subtitle Generation: Creates professional-grade .srt and .ass subtitle files with automatic line wrapping, duration management, and deduplication for readability.

Font-Aware Styling: Automatically selects appropriate fonts for different languages (e.g., Devanagari, Arabic, CJK) in the .ass subtitles to ensure correct rendering.

Video Processing: Burns the generated subtitles and dubbed audio into new video files using FFmpeg.

Robust & Resilient: Handles various media formats, splits large files for processing, and includes sophisticated logic to prevent common errors.

⚙️ How It Works: The Pipeline
The script follows a sequential AI pipeline to process the media:

Audio Extraction: FFmpeg extracts the audio from the input video into a standardized format.

Transcription (Speech-to-Text): The audio is transcribed into text with precise word timings using Whisper.

Translation: The transcribed text is translated segment by segment into the target language using the NLLB model.

Subtitle Creation: The translated text and original timestamps are used to generate a raw subtitle file (.srt), which is then refined for optimal line length and display time. An advanced .ass file is also created for styled rendering.

Dubbing (Text-to-Speech): The final translated text is synthesized into a complete audio track using an MMS TTS model for the target language. The audio is automatically paced to match the original video's duration.

Final Video Output: Two new video files are generated:

One with the original audio and burned-in translated subtitles.

One with the newly generated dubbed audio and burned-in translated subtitles.

🔧 Setup and Installation
Prerequisites
Python: Python 3.9 or newer is required.

FFmpeg: You must have FFmpeg installed on your system and accessible from your command line.

Download from the official website: https://ffmpeg.org/download.html

Installation Steps
Clone the Repository

git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
cd your-repository/backend

Create a Virtual Environment
It is highly recommended to use a virtual environment.

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies
Install all required Python packages from the requirements.txt file.

pip install -r requirements.txt

Note: This installation will download several large AI models (PyTorch, Transformers, etc.) and may take some time.

Download SpaCy Model
The script uses SpaCy for some text processing tasks. Download the required model:

python -m spacy download en_core_web_sm

🚀 Usage
Run the script from your terminal, providing the path to your input file and any desired options.

Basic Command
python subtitle.py [path_to_your_video.mp4] --target_lang [language_code]

Example
To process a video named sample_video.mp4 and translate it into Hindi, using the small Whisper model:

python subtitle.py sample_video.mp4 --model small --target_lang hin_Deva

Command-Line Arguments
Argument

Shorthand

Description

Default

Example

input



(Required) The path to the input video or audio file.

N/A

my_video.mp4

--model

-m

The Whisper model size. Options: tiny, base, small, medium, large. Smaller models are faster but less accurate.

large

--model small

--target_lang

-t

The target language for translation and dubbing. Use NLLB language codes.

hin_Deva

--target_lang spa_Latn

--language

-l

Force the source language of the audio (ISO 639-1 code). Bypasses auto-detection.

None

--language en

--no-dedupe



Disables the automatic deduplication of repeated words in the translation.

False

--no-dedupe

📁 Output Files
After a successful run, the script will generate the following files in the same directory:

your_video_[lang]_raw.srt: The initial, unrefined subtitle file.

your_video_[lang]_final.srt: The final, refined subtitle file, safe for all video players.

your_video_[lang]_final.ass: The advanced subtitle file with language-specific font styling.

your_video_[lang]_subs.mp4: A new video with the original audio and burned-in translated subtitles.

your_video_[lang]_tts.mp4: A new video with the dubbed audio and burned-in translated subtitles.

⚠️ Troubleshooting
MemoryError: Your computer ran out of RAM trying to load a large model.

Solution: Use a smaller model with the --model flag (e.g., --model medium or --model small).

OSError: [Errno 28] No space left on device: Your hard drive (usually the C: drive) is full. The script's cache and temporary files can consume significant space.

Solution:

Free up several gigabytes of space on your C: drive.

Delete the cache folder at C:\Users\YourUser\.cache\whisper to remove corrupted or incomplete model downloads.

AttributeError: module 'ffmpeg' has no attribute 'probe': A Python library conflict is occurring.

Solution: Reinstall the correct library by running these commands in order:

pip uninstall ffmpeg
pip uninstall ffmpeg-python
pip install ffmpeg-python

Compiler Errors on Windows (e.g., for monotonic_align): The installation fails because a C++ compiler is not set up correctly.

Solution: You must install packages that need compiling from the x64 Native Tools Command Prompt for VS. First, install Visual Studio Build Tools, and then use this special terminal to run the pip install -r requirements.txt command.