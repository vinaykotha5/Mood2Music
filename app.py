import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import tempfile
import subprocess
import atexit

# Optional: Move Hugging Face cache to D: drive
os.environ["HF_HOME"] = "D:/hf_cache"

# Set FFmpeg path (ensure FFmpeg is correctly installed)
os.environ["FFMPEG_BINARY"] = r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
FFMPEG_PATH = os.environ["FFMPEG_BINARY"]

# Function to check FFmpeg availability
def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_PATH, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        st.error("FFmpeg not found. Please ensure it is installed and the path is correctly set.")
        return False
    return True

# Load the MusicGen model
@st.cache_resource
def load_model():
    try:
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        model.set_generation_params(duration=10)  # Default
        return model
    except Exception as e:
        st.error(f"Error loading MusicGen model: {e}")
        return None

# Streamlit UI
model = load_model()

if model:
    st.title("🎵 AI Music Generator")

    st.markdown("Generate music from text prompts using [MusicGen](https://github.com/facebookresearch/audiocraft)!")

    use_custom_prompt = st.checkbox("Use custom prompt?")
    if use_custom_prompt:
        prompt = st.text_input("Enter your custom music prompt:", "an emotional orchestral soundtrack")
    else:
        mood = st.text_input("Enter Mood (e.g., happy, sad, energetic):", "happy")
        genre = st.text_input("Enter Genre (e.g., pop, rock, jazz):", "pop")
        prompt = f"{mood} {genre} music"

    duration = st.slider("Select Duration (seconds):", 5, 30, 10)

    if st.button("Generate Music"):
        if check_ffmpeg():
            with st.spinner("Generating music..."):
                try:
                    model.set_generation_params(duration=duration)
                    wav = model.generate([prompt])

                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                        base_path = fp.name.replace(".mp3", "")

                    audio_write(base_path, wav[0].cpu(), model.sample_rate, format="mp3")
                    output_path = base_path + ".mp3"

                    # Register temp file for cleanup on app close
                    atexit.register(lambda: os.remove(output_path) if os.path.exists(output_path) else None)

                    st.success("Music generated! 🎶")
                    st.audio(output_path)
                except Exception as e:
                    st.error(f"Error during music generation: {e}")
        else:
            st.error("FFmpeg is not available. Please check your installation and path settings.")
