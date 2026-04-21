---
title: AI Music Studio
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: aaa.py
pinned: false
license: mit
---

# 🎵 AI Music Studio

An AI-powered music generation and instrument conversion app built with [MusicGen](https://github.com/facebookresearch/audiocraft) (Facebook AI).

## Features

### 🎵 Generate Music
- Select genre, mood, instruments, tempo and duration
- Write custom prompts
- Animated vinyl disc player
- Waveform visualization + MP3 download

### 🎸 Tune Converter
Upload **any audio file** (MP3, WAV, OGG, FLAC) and convert it to a different instrument:
- Automatically detects **key, tempo, energy and brightness** using librosa
- Choose from **15 instruments**: Piano, Violin, Guitar, Flute, Saxophone, Trumpet, Harp, Cello, Synth, Marimba, Organ, Clarinet, Banjo, Ambient Pad...
- Choose from **8 styles**: Classical, Jazz, Electronic, Cinematic, Ambient, Folk, Pop, Blues
- Download converted MP3

### 📊 Visualizer
- Waveform comparison (original vs converted)
- Log-frequency spectrogram
- Chromagram (melody fingerprint)

## How Tune Conversion Works

1. Upload audio → librosa extracts **key, tempo, energy, brightness**
2. These features + your chosen instrument + style → rich text prompt
3. MusicGen generates audio matching the prompt
4. Result: same musical character, new instrument timbre

## Technical Notes

- Model: `facebook/musicgen-small` (~300MB)
- Runs on CPU (first generation takes 3–10 minutes)
- No GPU required
