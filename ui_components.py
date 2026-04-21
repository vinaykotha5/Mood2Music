import streamlit as st

def get_user_input():
    st.title("🎶 AI Music Generator with MusicGen")

    mood = st.selectbox("Select Mood", ["Happy", "Sad", "Chill", "Epic", "Energetic"])
    genre = st.selectbox("Select Genre", ["Classical", "Jazz", "EDM", "Rock", "Ambient"])
    duration = st.slider("Duration (seconds)", 5, 30, 10)
    generate_btn = st.button("Generate Music")

    return mood, genre, duration, generate_btn
