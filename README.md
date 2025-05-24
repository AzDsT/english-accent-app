# 🎙️ English Accent Classifier & Summarizer

This Streamlit app accepts a public video URL, extracts the audio, identifies the speaker's English accent, transcribes the speech, and generates a short summary. It's designed to help evaluate spoken English for hiring and review purposes.

## 🔧 Features

- 📥 Accepts direct MP4 video URLs
- 🔊 Extracts audio using `moviepy` and `pydub`
- 🧠 Transcribes speech using OpenAI `whisper`
- 🌍 Detects English accents (US, UK, Indian, etc.) using a pre-trained SpeechBrain model
- 📝 Summarizes transcript using `facebook/bart-large-cnn`
- 📄 Transcript available for download
- 📊 Audio waveform visualization

## 🚀 How to Run

1. Clone or download this repository.
2. Open your terminal in the project folder.
3. Install the required dependencies:

pip install -r requirements.txt
streamlit run app.py


## 📺 Demo & Test
- 🎥 [Watch the Loom Video Walkthrough](https://www.loom.com/share/a0e1d6c133ee41daa80196b638432404?sid=16597d24-1b15-4cdc-a959-fa687d2bb203)
- 🔗 [Sample MP4 Test Video](https://files.catbox.moe/5oteou.mp4)
