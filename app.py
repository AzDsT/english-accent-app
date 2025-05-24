from accent_classifier import classify_accent
import streamlit as st
import requests
import whisper
from moviepy import VideoFileClip
from transformers import pipeline
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt

def download_video(url, output_path="input_video.mp4"):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, stream=True, headers=headers)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path

def extract_audio(video_path, output_audio="output_audio.wav"):
    audio = AudioSegment.from_file(video_path)
    audio.export(output_audio, format="wav")
    return output_audio

def plot_waveform(audio_path):
    waveform, sample_rate = sf.read(audio_path)
    plt.figure(figsize=(10, 2))
    plt.plot(waveform)
    plt.title("Audio Waveform")
    st.pyplot(plt)

def detect_english_fluency(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    language = result.get("language")
    transcript = result.get("text", "")
    accent, confidence = classify_accent(audio_path)
    return language, accent, confidence, transcript

# Streamlit UI
st.title("🎙️ English Accent & Summary Tool")
video_url = st.text_input("Paste a public video URL (MP4 link):")

if video_url:
    with st.spinner("📥 Downloading video..."):
        video_path = download_video(video_url)
    
    with st.spinner("🔊 Extracting audio..."):
        audio_path = extract_audio(video_path)

    with st.spinner("📊 Visualizing waveform..."):
        plot_waveform(audio_path)
    
    with st.spinner("🧠 Transcribing and analyzing..."):
        lang, accent, conf, transcript = detect_english_fluency(audio_path)

    with st.spinner("📝 Summarizing transcript..."):
        if len(transcript.strip().split()) > 10:  # Require at least 10 words
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            try:
                summary = summarizer(transcript, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                summary = f"Summarization failed: {e}"
        else:
            summary = "Transcript is too short to summarize."


    st.success("✅ Analysis Complete!")
    st.markdown(f"**🌍 Detected Language:** `{lang}`")
    st.markdown(f"**🗣️ Accent:** `{accent}`")
    st.markdown(f"**🎯 Confidence Score:** `{conf}`")
    
    st.markdown("**📄 Transcript:**")
    st.write(transcript)

    st.markdown("**🧾 Summary:**")
    st.write(summary)

    st.download_button("📥 Download Transcript", transcript, file_name="transcript.txt")
