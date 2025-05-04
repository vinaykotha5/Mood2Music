import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import tempfile
import os
import time
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Config
os.environ["HF_HOME"] = "D:/hf_cache"
FFMPEG_PATH = r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# Session state
def init_session_state():
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'generated_music_path' not in st.session_state:
        st.session_state.generated_music_path = None

def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_prompt(genres, mood, instruments, tempo):
    return f"{', '.join(genres)} music with {', '.join(instruments)}, mood {mood}, tempo {tempo} bpm"

def get_rotating_disc_html(is_playing):
    return f"""
        <div class="vinyl-player" style="width: 300px; height: 300px; margin: 0 auto; position: relative;">
            <style>
                @keyframes rotate {{
                    from {{ transform: rotate(0deg); }}
                    to {{ transform: rotate(360deg); }}
                }}
                .vinyl-disc {{
                    width: 100%;
                    height: 100%;
                    background: #1a1a1a;
                    border-radius: 50%;
                    position: relative;
                    box-shadow: 0 0 10px rgba(0,0,0,0.5);
                    {'animation: rotate 3s linear infinite;' if is_playing else ''}
                }}
                .vinyl-label {{
                    width: 40%;
                    height: 40%;
                    background: #4a5568;
                    border-radius: 50%;
                    position: absolute;
                    top: 30%;
                    left: 30%;
                    border: 2px solid #2d3748;
                }}
                .vinyl-hole {{
                    width: 10%;
                    height: 10%;
                    background: #1a1a1a;
                    border-radius: 50%;
                    position: absolute;
                    top: 45%;
                    left: 45%;
                    border: 1px solid #333;
                }}
                .vinyl-grooves {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: repeating-radial-gradient(
                        circle at 50% 50%,
                        transparent 0,
                        transparent 2px,
                        rgba(51, 51, 51, 0.3) 3px,
                        transparent 4px
                    );
                    border-radius: 50%;
                }}
                .spiral {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: conic-gradient(
                        from 0deg,
                        transparent 0deg,
                        rgba(51, 51, 51, 0.2) 90deg,
                        transparent 180deg,
                        rgba(51, 51, 51, 0.2) 270deg,
                        transparent 360deg
                    );
                    border-radius: 50%;
                }}
                .arm {{
                    width: 100px;
                    height: 8px;
                    background: #666;
                    position: absolute;
                    top: 20%;
                    right: -20px;
                    transform: rotate(-20deg);
                    transform-origin: right center;
                    border-radius: 4px;
                }}
                .needle {{
                    width: 4px;
                    height: 15px;
                    background: #444;
                    position: absolute;
                    bottom: -15px;
                    right: 0;
                    border-radius: 2px;
                }}
                .play-status {{
                    position: absolute;
                    top: -30px;
                    left: 50%;
                    transform: translateX(-50%);
                    color: #4CAF50;
                    font-weight: bold;
                    font-family: Arial, sans-serif;
                }}
            </style>
            <div class="play-status">{("♫ Now Playing" if is_playing else "● Paused")}</div>
            <div class="vinyl-disc">
                <div class="vinyl-grooves"></div>
                <div class="spiral"></div>
                <div class="vinyl-label"></div>
                <div class="vinyl-hole"></div>
            </div>
            <div class="arm">
                <div class="needle"></div>
            </div>
        </div>
    """

def generate_music(prompt, duration):
    model = load_model()
    model.set_generation_params(duration=duration)
    start_time = time.time()
    wav = model.generate([prompt])
    generation_time = time.time() - start_time

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        base = fp.name.replace(".mp3", "")
        audio_write(base, wav[0].cpu(), model.sample_rate, format="mp3")
        return base + ".mp3", wav[0].cpu().numpy(), model.sample_rate, generation_time
def plot_waveform(wav_data, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(wav_data, sr=sr, ax=ax)
    ax.set_title("Generated Melody Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


# Streamlit App
def main():
    st.set_page_config(page_title="🎵 AI Music Generator", layout="wide")
    init_session_state()
    
    st.title("🎵 AI Music Generator")
    st.markdown("Craft your own AI-generated music with mood, instruments, and tempo!")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.header("Music Parameters")
        genres = st.multiselect("Select Genre(s)", ["Classical", "Jazz", "Rock", "Electronic", "Folk", "Pop"], default=["Pop"])
        mood = st.select_slider("Mood", ["Very Calm", "Calm", "Neutral", "Energetic", "Very Energetic"], value="Neutral")
        instruments = st.multiselect("Instruments", ["Piano", "Violin", "Guitar", "Drums"], default=["Piano"])
        tempo = st.slider("Tempo (BPM)", 40, 160, 100)
        duration = st.slider("Duration (seconds)", 10, 30, 10)
    

        if st.button("Generate Music"):
            with st.spinner("Generating..."):
                progress = st.progress(0)
                for i in range(30):  # Simulate loading
                    time.sleep(0.05)
                    progress.progress(int((i+1)*100/30))
                prompt = generate_prompt(genres, mood, instruments, tempo)
                try:
                    output_path, wav_data, sr, generation_time = generate_music(prompt, duration)
                    st.session_state.generated_music_path = output_path
                    st.session_state.is_playing = True
                    st.success(f"Music generated in {generation_time:.2f} seconds")
                    plot_waveform(wav_data, sr)

                except Exception as e:
                    st.error(f"Music generation failed: {e}")

    with col2:
        st.header("MP3 Player")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Play" if not st.session_state.is_playing else "⏸ Pause", on_click=toggle_play):
                st.rerun()
        st.components.v1.html(get_rotating_disc_html(st.session_state.is_playing), height=350)

    with col3:
        st.header("Controls")
        if st.session_state.generated_music_path:
            with open(st.session_state.generated_music_path, "rb") as f:
                audio_data = f.read()
                st.audio(audio_data, format="audio/mp3")
                st.download_button("Download MP3", data=audio_data, file_name="generated_music.mp3", mime="audio/mpeg")

if __name__ == "__main__":
    main()
