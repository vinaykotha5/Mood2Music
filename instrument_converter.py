"""
instrument_converter.py
────────────────────────
Helper module for the Tune Converter tab.
"""

# ── MUST be first: installs a TF stub so torch.utils.tensorboard doesn't crash
import tensorflow_mock  # noqa: F401

import os
import io
import time
import tempfile

import numpy as np
import librosa
import streamlit as st
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


# ─── Instrument → base prompt ────────────────────────────────────────────────

INSTRUMENT_PROMPTS: dict[str, str] = {
    "🎹 Piano":            "solo piano, clean acoustic tone, melodic and expressive",
    "🎻 Violin":           "solo violin, orchestral strings, lyrical and expressive melody",
    "🎸 Acoustic Guitar":  "acoustic guitar fingerpicking, warm wooden tone, melodic",
    "🎸 Electric Guitar":  "electric guitar lead melody, clean crisp tone",
    "🪈 Flute":            "solo flute, classical, airy, light and delicate melody",
    "🎷 Saxophone":        "jazz saxophone, smooth and warm, soulful melody",
    "🎺 Trumpet":          "solo trumpet, bright bold brass melody",
    "🎸 Banjo":            "banjo melody, Appalachian folk style, plucked strings",
    "🎼 Cello":            "solo cello, deep emotional strings, rich resonant tone",
    "🎵 Harp":             "harp melody, ethereal flowing arpeggios, delicate",
    "🎹 Synth Lead":       "synthesizer lead melody, electronic, punchy and bright",
    "🌊 Ambient Pad":      "ambient synth pad, atmospheric, cinematic texture",
    "🪘 Marimba":          "marimba melody, bright mallet percussion, Latin flavour",
    "⛪ Organ":             "church pipe organ, grand and majestic, classical",
    "🎻 Clarinet":         "solo clarinet, warm woodwind tone, classical melody",
}

# ─── Style modifiers ─────────────────────────────────────────────────────────

STYLE_MODIFIERS: dict[str, str] = {
    "Classical":    "classical composition, formal, orchestral arrangement",
    "Jazz":         "jazz improvisation, swing rhythm, blue notes",
    "Electronic":   "electronic music production, synthesised textures, modern",
    "Cinematic":    "cinematic film score, emotional, epic and adventurous",
    "Ambient":      "ambient meditative, spacious, calm and minimal",
    "Folk":         "folk acoustic, natural, rustic and traditional",
    "Pop":          "pop production, catchy, bright and polished",
    "Blues":        "blues feeling, soulful, raw and expressive",
}

# ─── Constants ───────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─── Model loading (cached so it only loads once) ─────────────────────────────

@st.cache_resource(show_spinner=False)
def load_gen_model() -> MusicGen:
    """Load MusicGen-small on CPU (cached across Streamlit re-runs)."""
    import platform
    if platform.system() == "Windows":
        os.environ.setdefault("HF_HOME", "D:/hf_cache")
    else:
        os.environ.setdefault("HF_HOME", "/data/hf_cache")
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.lm.to("cpu")
    return model


# ─── Audio analysis ──────────────────────────────────────────────────────────

def analyze_audio(audio_bytes: bytes) -> dict:
    """
    Analyse an uploaded audio file and return a dictionary of musical features.
    Returns a dict with keys: tempo, key, mode, energy, duration, waveform, sample_rate.
    On error, returns {"error": <message>}.
    """
    tmp_path = None
    try:
        # Write bytes to a temp file so librosa can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None, mono=True)

        # ── Tempo ────────────────────────────────────────────────────────────
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo_arr)[0])
        if tempo < 20 or tempo > 300:      # fallback for unreliable detections
            tempo = 100.0

        # ── Key / scale ──────────────────────────────────────────────────────
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)           # shape (12,)
        key_idx = int(np.argmax(chroma_mean))
        detected_key = NOTE_NAMES[key_idx]

        # Major vs minor heuristic (compare major/minor profile correlations)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_norm = chroma_mean / (chroma_mean.max() + 1e-8)
        maj_corr = np.corrcoef(chroma_norm, np.roll(major_profile / major_profile.max(), key_idx))[0, 1]
        min_corr = np.corrcoef(chroma_norm, np.roll(minor_profile / minor_profile.max(), key_idx))[0, 1]
        mode = "major" if maj_corr >= min_corr else "minor"

        # ── Energy / mood ────────────────────────────────────────────────────
        rms = float(librosa.feature.rms(y=y).mean())
        if rms > 0.12:
            energy = "energetic and dynamic"
        elif rms > 0.05:
            energy = "moderately energetic"
        else:
            energy = "calm and gentle"

        # ── Spectral brightness (helps further colour the prompt) ─────────────
        cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        brightness = "bright and high-register" if cent > 3000 else "warm and mid-register"

        duration = float(librosa.get_duration(y=y, sr=sr))

        return {
            "tempo":       tempo,
            "key":         detected_key,
            "mode":        mode,
            "energy":      energy,
            "brightness":  brightness,
            "duration":    duration,
            "waveform":    y,
            "sample_rate": sr,
        }

    except Exception as exc:
        return {"error": str(exc)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─── Prompt builder ──────────────────────────────────────────────────────────

def build_conversion_prompt(instrument: str, style: str, analysis: dict) -> str:
    """
    Combine instrument base prompt + style modifier + audio analysis features
    into a single rich text prompt for MusicGen.
    """
    base  = INSTRUMENT_PROMPTS.get(instrument, "solo piano melody")
    style_mod = STYLE_MODIFIERS.get(style, "")

    key        = analysis.get("key",        "C")
    mode       = analysis.get("mode",       "major")
    tempo      = analysis.get("tempo",      100.0)
    energy     = analysis.get("energy",     "moderate")
    brightness = analysis.get("brightness", "warm and mid-register")

    prompt = (
        f"{base}, {key} {mode} key, approximately {int(tempo)} bpm, "
        f"{energy}, {brightness}, {style_mod}"
    )
    return prompt


# ─── Main conversion entry-point ─────────────────────────────────────────────

def convert_tune(
    audio_bytes: bytes,
    instrument:  str,
    style:       str,
    duration:    int,
) -> tuple:
    """
    Analyse *audio_bytes*, build a prompt, generate with MusicGen-small.

    Returns:
        (output_mp3_path, wav_np, sample_rate, analysis_dict, prompt_str)
        On error: (None, None, None, None, error_message)
    """
    # 1. Analyse
    analysis = analyze_audio(audio_bytes)
    if "error" in analysis:
        return None, None, None, None, f"Analysis error: {analysis['error']}"

    # 2. Build prompt
    prompt = build_conversion_prompt(instrument, style, analysis)

    # 3. Generate
    try:
        model = load_gen_model()
        model.set_generation_params(duration=min(int(duration), 30))

        start = time.time()
        with torch.no_grad():
            try:
                wav_tensor = model.generate([prompt])          # shape (1, 1, T)
            except TypeError as te:
                # If isinstance error, try to provide more context
                if "isinstance() arg 2" in str(te):
                    import traceback
                    tb = traceback.format_exc()
                    return None, None, None, None, f"Type checking error in audiocraft. This may be due to mock dependencies. Full error: {tb}"
                raise
        elapsed = time.time() - start

        wav_cpu = wav_tensor[0].cpu()                     # shape (1, T)
        wav_np  = wav_cpu.squeeze(0).numpy()              # shape (T,)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            base = fp.name.replace(".mp3", "")

        audio_write(base, wav_cpu, model.sample_rate, format="mp3",
                    strategy="loudness")

        analysis["generation_time"] = elapsed
        return base + ".mp3", wav_np, model.sample_rate, analysis, prompt

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return None, None, None, None, f"Generation error: {exc}\n\nTraceback:\n{tb}"
