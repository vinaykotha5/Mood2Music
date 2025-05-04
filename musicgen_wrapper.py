import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import tempfile

# Set ffmpeg path manually
os.environ["FFMPEG_BINARY"] = r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# Load the model
@st.cache_resource
def load_model():
    try:
        model = MusicGen.get_pretrained('D:\\vs code\\major0\\Music-Generation-Using-Deep-Learning-master')
        model.set_generation_params(duration=10)  # Default duration
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Streamlit UI
st.title("🎵 AI Music Generator")

mood = st.text_input("Enter Mood (e.g., happy, sad, energetic):", "happy")
genre = st.text_input("Enter Genre (e.g., pop, rock, jazz):", "pop")
duration = st.slider("Select Duration (seconds):", 5, 30, 10)

if st.button("Generate Music"):
    if model:
        with st.spinner("Generating music... please wait ⏳"):
            try:
                prompt = f"{mood} {genre} music"
                model.set_generation_params(duration=duration)
                wav = model.generate([prompt])
                
                # Save audio
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    output_path = fp.name
                audio_write(output_path.replace(".mp3", ""), wav[0].cpu(), model.sample_rate, format="mp3")

                # Display audio
                st.success("Music generated successfully! 🎶")
                st.audio(output_path)
            except Exception as e:
                st.error(f"Error during music generation: {e}")
    else:
        st.error("Model is not loaded. Please check the logs for details.")
