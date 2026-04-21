"""
aaa.py  —  AI Music Studio
───────────────────────────
Multi-tab Streamlit app:
  Tab 1 ▸ 🎵 Generate Music   – prompt-based generation (existing feature, polished)
  Tab 2 ▸ 🎸 Tune Converter   – upload any audio → re-render as chosen instrument
  Tab 3 ▸ 📊 Visualizer       – waveform / spectrogram / chromagram analysis

Optimised for Intel integrated graphics (CPU-only inference).
"""

# ── MUST be first: installs a TF stub so torch.utils.tensorboard doesn't crash
import tensorflow_mock  # noqa: F401

import os
import io
import time
import tempfile
import atexit

import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from instrument_converter import (
    analyze_audio,
    convert_tune,
    load_gen_model,
    INSTRUMENT_PROMPTS,
    STYLE_MODIFIERS,
)

import platform as _platform

# ─── Environment (cross-platform: Windows local + Linux HF Spaces) ───────────

if _platform.system() == "Windows":
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    _win_ffmpeg = r"D:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(_win_ffmpeg):
        os.environ["PATH"] += os.pathsep + os.path.dirname(_win_ffmpeg)
else:
    # Linux (Hugging Face Spaces) — ffmpeg installed via packages.txt
    os.environ.setdefault("HF_HOME", "/data/hf_cache")


# ─── Session state ───────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "is_playing":          False,
        "generated_music_path": None,
        "gen_wav_data":         None,
        "gen_sr":               None,
        "converted_music_path": None,
        "conv_wav_data":        None,
        "conv_sr":              None,
        "orig_wav_data":        None,
        "orig_sr":              None,
        "audio_analysis":       None,
        "conversion_prompt":    None,
        "uploaded_bytes":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing

# ─── CSS & theming ───────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #080812 0%, #0f0f1f 40%, #12122a 100%);
    min-height: 100vh;
}

/* ── Header ── */
.studio-header {
    text-align: center;
    padding: 2.2rem 0 0.4rem;
}
.studio-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a855f7, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin: 0;
}
.studio-sub {
    color: #64748b;
    font-size: 1rem;
    margin: 0.3rem 0 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.92rem;
    border: none;
    background: transparent;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(59,130,246,0.25)) !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(168,85,247,0.4) !important;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.accent-card {
    background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(59,130,246,0.08));
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.teal-card {
    background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(16,185,129,0.07));
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

/* ── Metric chips ── */
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.6rem 0; }
.metric-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}
.metric-chip span { color: #a78bfa; font-weight: 600; }

/* ── Prompt box ── */
.prompt-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #c4b5fd;
    font-style: italic;
    word-break: break-word;
    margin-top: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2.2rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(124,58,237,0.45) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #065f46, #047857) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Sliders / inputs ── */
.stSlider [data-testid="stTickBar"] { color: #64748b; }
label { color: #94a3b8 !important; font-size: 0.88rem !important; }

/* ── Section headers ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Warning banner ── */
.cpu-banner {
    background: rgba(234,179,8,0.1);
    border: 1px solid rgba(234,179,8,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.83rem;
    color: #fde68a;
    margin-bottom: 1rem;
}

/* ── Audio ── */
audio { width: 100%; border-radius: 10px; margin-top: 0.5rem; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
"""

# ─── Vinyl disc animation ─────────────────────────────────────────────────────

def vinyl_html(is_playing: bool) -> str:
    anim = "animation: rotate 3s linear infinite;" if is_playing else ""
    status = "♫ Now Playing" if is_playing else "● Ready"
    status_color = "#4ade80" if is_playing else "#94a3b8"
    return f"""
    <div style="width:260px;height:290px;margin:0 auto;position:relative;padding-top:30px;">
      <style>
        @keyframes rotate {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
        .vd{{ width:240px;height:240px;background:radial-gradient(circle,#1e1e3a,#0f0f1a);
              border-radius:50%;position:relative;box-shadow:0 0 40px rgba(124,58,237,0.3);
              {anim} }}
        .vg{{ position:absolute;top:0;left:0;width:100%;height:100%;border-radius:50%;
              background:repeating-radial-gradient(circle at 50% 50%,transparent 0,transparent 5px,
              rgba(168,85,247,0.07) 6px,transparent 7px); }}
        .vl{{ width:38%;height:38%;background:linear-gradient(135deg,#3b1f7a,#1e3a8a);
              border-radius:50%;position:absolute;top:31%;left:31%;
              border:2px solid rgba(168,85,247,0.5);
              box-shadow:0 0 20px rgba(124,58,237,0.5); }}
        .vh{{ width:9%;height:9%;background:#080812;border-radius:50%;
              position:absolute;top:45.5%;left:45.5%; }}
        .arm{{ width:90px;height:7px;background:linear-gradient(90deg,#4b5563,#9ca3af);
               position:absolute;top:22%;right:-15px;transform:rotate(-25deg);
               transform-origin:right center;border-radius:4px; }}
        .needle{{ width:3px;height:14px;background:#6b7280;position:absolute;
                  bottom:-14px;right:0;border-radius:2px; }}
      </style>
      <div style="text-align:center;color:{status_color};font-weight:600;
                  font-size:0.85rem;margin-bottom:8px;">{status}</div>
      <div class="vd">
        <div class="vg"></div>
        <div class="vl"></div>
        <div class="vh"></div>
      </div>
      <div class="arm"><div class="needle"></div></div>
    </div>
    """

# ─── Chart helpers ────────────────────────────────────────────────────────────

CHART_BG  = "#0a0a1a"
CHART_FG  = "#12122a"
TITLE_CLR = "#e2e8f0"
TICK_CLR  = "#64748b"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CHART_FG)
    ax.set_title(title,  color=TITLE_CLR, fontsize=11, fontweight="600", pad=8)
    ax.set_xlabel(xlabel, color=TICK_CLR,  fontsize=9)
    ax.set_ylabel(ylabel, color=TICK_CLR,  fontsize=9)
    ax.tick_params(colors=TICK_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
        spine.set_color("#1e293b")

def plot_waveform_compare(orig_y, orig_sr, conv_y, conv_sr):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5))
    fig.patch.set_facecolor(CHART_BG)
    librosa.display.waveshow(orig_y, sr=orig_sr, ax=axes[0], color="#a855f7", alpha=0.85)
    _style_ax(axes[0], "Original Tune — Waveform", "Time (s)", "Amplitude")
    librosa.display.waveshow(conv_y, sr=conv_sr, ax=axes[1], color="#06b6d4", alpha=0.85)
    _style_ax(axes[1], "Converted Tune — Waveform", "Time (s)", "Amplitude")
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)

def plot_spectrogram(y, sr, title, color="magma"):
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor(CHART_BG)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap=color)
    fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(colors=TICK_CLR, labelsize=7)
    _style_ax(ax, title, "Time (s)", "Frequency (Hz)")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close(fig)

def plot_chromagram(y, sr, title):
    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor(CHART_BG)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma",
                                    ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax).ax.tick_params(colors=TICK_CLR, labelsize=7)
    _style_ax(ax, title, "Time (s)", "Pitch Class")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close(fig)

def plot_single_waveform(y, sr, title, color="#a855f7"):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(CHART_BG)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=color, alpha=0.85)
    _style_ax(ax, title, "Time (s)", "Amplitude")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close(fig)

# ─── Generate tab helpers ─────────────────────────────────────────────────────

def generate_music_from_prompt(prompt: str, duration: int):
    model = load_gen_model()
    model.set_generation_params(duration=duration)
    start = time.time()
    with torch.no_grad():
        wav = model.generate([prompt])
    elapsed = time.time() - start
    wav_cpu = wav[0].cpu()
    wav_np  = wav_cpu.squeeze(0).numpy()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        base = fp.name.replace(".mp3", "")
    audio_write(base, wav_cpu, model.sample_rate, format="mp3", strategy="loudness")
    return base + ".mp3", wav_np, model.sample_rate, elapsed

# ─── TAB 1 — Generate Music ───────────────────────────────────────────────────

def tab_generate():
    st.markdown('<div class="cpu-banner">⚡ Running on CPU — generation takes ~2–5 min. Please be patient!</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-header">🎛️ Music Parameters</div>', unsafe_allow_html=True)
        with st.container():
            genres = st.multiselect(
                "Genre(s)", ["Classical", "Jazz", "Rock", "Electronic", "Folk", "Pop", "Blues", "Ambient"],
                default=["Pop"],
            )
            mood = st.select_slider(
                "Mood",
                options=["Very Calm", "Calm", "Neutral", "Energetic", "Very Energetic"],
                value="Neutral",
            )
            instruments = st.multiselect(
                "Instruments",
                ["Piano", "Violin", "Guitar", "Drums", "Flute", "Saxophone", "Synth"],
                default=["Piano"],
            )
            col_t, col_d = st.columns(2)
            with col_t:
                tempo = st.slider("Tempo (BPM)", 40, 200, 100)
            with col_d:
                duration = st.slider("Duration (s)", 5, 30, 10)

            use_custom = st.toggle("✏️ Write custom prompt")
            if use_custom:
                custom_prompt = st.text_area(
                    "Custom prompt",
                    value="an emotional orchestral piece with piano and strings",
                    height=80,
                )
                final_prompt = custom_prompt
            else:
                final_prompt = (
                    f"{', '.join(genres) if genres else 'pop'} music with "
                    f"{', '.join(instruments) if instruments else 'piano'}, "
                    f"mood {mood}, tempo {tempo} bpm"
                )
                st.markdown(f'<div class="prompt-box">📝 {final_prompt}</div>', unsafe_allow_html=True)

            generate_btn = st.button("🎵 Generate Music", key="gen_btn", use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🎧 Player</div>', unsafe_allow_html=True)
        st.components.v1.html(vinyl_html(st.session_state.is_playing), height=310)
        if st.session_state.is_playing:
            st.button("⏸ Pause", on_click=toggle_play, key="pause_btn", use_container_width=True)
        else:
            st.button("▶ Play", on_click=toggle_play, key="play_btn", use_container_width=True)

        if st.session_state.generated_music_path and os.path.exists(st.session_state.generated_music_path):
            with open(st.session_state.generated_music_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download MP3",
                data=audio_bytes,
                file_name="generated_music.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )

    if generate_btn:
        with st.spinner("🎼 Generating music on CPU… this may take a few minutes."):
            prog = st.progress(0, text="Initialising model…")
            for i in range(20):
                time.sleep(0.06)
                prog.progress(int((i + 1) * 100 / 20), text="Loading model…")
            try:
                out_path, wav_np, sr, elapsed = generate_music_from_prompt(final_prompt, duration)
                st.session_state.generated_music_path = out_path
                st.session_state.gen_wav_data = wav_np
                st.session_state.gen_sr = sr
                st.session_state.is_playing = True
                prog.progress(100, text="Done!")
                st.success(f"✅ Generated in {elapsed:.0f} seconds")
                atexit.register(lambda p=out_path: os.remove(p) if os.path.exists(p) else None)
                st.rerun()
            except Exception as exc:
                prog.empty()
                st.error(f"❌ Generation failed: {exc}")

    # Waveform of generated audio
    if st.session_state.gen_wav_data is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Waveform</div>', unsafe_allow_html=True)
        plot_single_waveform(st.session_state.gen_wav_data, st.session_state.gen_sr,
                             "Generated Music — Waveform", color="#a855f7")


# ─── TAB 2 — Tune Converter ───────────────────────────────────────────────────

def tab_converter():
    st.markdown('<div class="cpu-banner">⚡ CPU mode — conversion analyses your tune then generates a new one. Expect ~2–5 min.</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
      <b style="color:#e2e8f0;">How it works</b>
      <p style="color:#94a3b8;font-size:0.88rem;margin:0.4rem 0 0;">
        Upload any audio file. The app detects its <b>key, tempo, mood and brightness</b>,
        then uses that musical DNA to guide MusicGen into re-creating the feel of your
        tune played by the chosen instrument. The result is a creative reinterpretation —
        same musical character, brand-new timbre.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # ── Upload ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📂 Upload Your Tune</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Accepts MP3, WAV, OGG, FLAC",
            type=["mp3", "wav", "ogg", "flac"],
            key="tune_upload",
        )

        if uploaded:
            audio_bytes = uploaded.read()
            st.session_state.uploaded_bytes = audio_bytes
            st.markdown("**Preview — Original:**")
            st.audio(audio_bytes, format=f"audio/{uploaded.name.split('.')[-1]}")

            # Auto-analyse
            with st.spinner("🔍 Analysing audio…"):
                analysis = analyze_audio(audio_bytes)

            if "error" not in analysis:
                st.session_state.audio_analysis = analysis
                st.session_state.orig_wav_data = analysis["waveform"]
                st.session_state.orig_sr       = analysis["sample_rate"]

                st.markdown('<div class="teal-card">', unsafe_allow_html=True)
                st.markdown("**🔬 Detected Musical Features**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🎵 Key",   f"{analysis['key']} {analysis['mode']}")
                m2.metric("🥁 Tempo", f"{analysis['tempo']:.0f} BPM")
                m3.metric("⚡ Energy", analysis["energy"].split()[0].capitalize())
                m4.metric("⏱ Length", f"{analysis['duration']:.1f}s")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"Analysis error: {analysis['error']}")

        st.markdown("---")

        # ── Instrument selection ─────────────────────────────────────────────
        st.markdown('<div class="section-header">🎸 Choose Target Instrument</div>', unsafe_allow_html=True)
        instrument = st.selectbox(
            "Instrument",
            options=list(INSTRUMENT_PROMPTS.keys()),
            index=0,
            key="instrument_select",
        )

        # ── Style ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🎨 Style</div>', unsafe_allow_html=True)
        style_cols = st.columns(4)
        style_options = list(STYLE_MODIFIERS.keys())
        style_choice = st.radio(
            "Style",
            options=style_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        out_duration = st.slider("Output Duration (s)", 5, 30, 15, key="conv_duration")

    with col2:
        st.markdown('<div class="section-header">🎧 Output</div>', unsafe_allow_html=True)

        if st.session_state.converted_music_path and os.path.exists(st.session_state.converted_music_path):
            with open(st.session_state.converted_music_path, "rb") as f:
                conv_bytes = f.read()
            st.markdown("**Converted Audio:**")
            st.audio(conv_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download Converted MP3",
                data=conv_bytes,
                file_name="converted_tune.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )

            if st.session_state.conversion_prompt:
                st.markdown("**Prompt used:**")
                st.markdown(
                    f'<div class="prompt-box">{st.session_state.conversion_prompt}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#475569;">
              <div style="font-size:3rem;">🎼</div>
              <p>Converted audio will appear here</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Convert button ───────────────────────────────────────────────────────
    convert_disabled = st.session_state.uploaded_bytes is None
    if convert_disabled:
        st.info("⬆️ Upload an audio file above to enable conversion.")

    if st.button("🎸 Convert Tune", key="convert_btn",
                 use_container_width=True, disabled=convert_disabled):
        with st.spinner(f"🎵 Converting to {instrument}… please wait (CPU mode)."):
            prog = st.progress(0, text="Building musical prompt…")
            prog.progress(15, text="Analysing key, tempo, mood…")
            time.sleep(0.3)
            prog.progress(30, text="Generating — this takes a few minutes on CPU…")

            out_path, wav_np, sr, analysis, result = convert_tune(
                st.session_state.uploaded_bytes,
                instrument,
                style_choice,
                out_duration,
            )

            if out_path:
                prog.progress(100, text="Done!")
                st.session_state.converted_music_path = out_path
                st.session_state.conv_wav_data         = wav_np
                st.session_state.conv_sr               = sr
                st.session_state.audio_analysis        = analysis
                st.session_state.conversion_prompt     = result
                elapsed = analysis.get("generation_time", 0)
                st.success(f"✅ Converted in {elapsed:.0f}s  |  Instrument: {instrument}  |  Style: {style_choice}")
                atexit.register(lambda p=out_path: os.remove(p) if os.path.exists(p) else None)
                st.rerun()
            else:
                prog.empty()
                st.error(f"❌ Conversion failed: {result}")


# ─── TAB 3 — Visualizer ───────────────────────────────────────────────────────

def tab_visualizer():
    has_orig = st.session_state.orig_wav_data is not None
    has_conv = st.session_state.conv_wav_data is not None
    has_gen  = st.session_state.gen_wav_data  is not None

    if not has_orig and not has_gen:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;color:#475569;">
          <div style="font-size:4rem;">📊</div>
          <h3 style="color:#64748b;">No audio to visualise yet</h3>
          <p>Generate music or convert a tune first, then come back here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    view = st.radio(
        "View",
        options=["Waveform", "Spectrogram", "Chromagram"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Waveform comparison ──────────────────────────────────────────────────
    if view == "Waveform":
        if has_orig and has_conv:
            st.markdown('<div class="section-header">🔊 Original vs Converted — Waveforms</div>',
                        unsafe_allow_html=True)
            plot_waveform_compare(
                st.session_state.orig_wav_data, st.session_state.orig_sr,
                st.session_state.conv_wav_data, st.session_state.conv_sr,
            )
        elif has_orig:
            st.markdown('<div class="section-header">🔊 Uploaded Tune — Waveform</div>',
                        unsafe_allow_html=True)
            plot_single_waveform(st.session_state.orig_wav_data, st.session_state.orig_sr,
                                 "Original Tune", "#a855f7")
        if has_gen:
            st.markdown('<div class="section-header">🎵 Generated Music — Waveform</div>',
                        unsafe_allow_html=True)
            plot_single_waveform(st.session_state.gen_wav_data, st.session_state.gen_sr,
                                 "Generated Music", "#06b6d4")

    # ── Spectrogram ──────────────────────────────────────────────────────────
    elif view == "Spectrogram":
        if has_orig:
            st.markdown('<div class="section-header">📡 Original Tune — Spectrogram</div>',
                        unsafe_allow_html=True)
            plot_spectrogram(st.session_state.orig_wav_data, st.session_state.orig_sr,
                             "Original — Log-frequency Spectrogram", "magma")
        if has_conv:
            st.markdown('<div class="section-header">📡 Converted Tune — Spectrogram</div>',
                        unsafe_allow_html=True)
            plot_spectrogram(st.session_state.conv_wav_data, st.session_state.conv_sr,
                             "Converted — Log-frequency Spectrogram", "viridis")
        if has_gen:
            st.markdown('<div class="section-header">📡 Generated Music — Spectrogram</div>',
                        unsafe_allow_html=True)
            plot_spectrogram(st.session_state.gen_wav_data, st.session_state.gen_sr,
                             "Generated — Log-frequency Spectrogram", "plasma")

    # ── Chromagram ───────────────────────────────────────────────────────────
    elif view == "Chromagram":
        st.markdown(
            '<div class="section-header">🎼 Chromagram — Pitch-class energy over time</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<p style="color:#64748b;font-size:0.85rem;">Brighter = stronger presence of that pitch class. '
                    'This is the melody "fingerprint" used to guide conversion.</p>', unsafe_allow_html=True)
        if has_orig:
            plot_chromagram(st.session_state.orig_wav_data, st.session_state.orig_sr,
                            "Original Tune — Chromagram")
        if has_conv:
            plot_chromagram(st.session_state.conv_wav_data, st.session_state.conv_sr,
                            "Converted Tune — Chromagram")
        if has_gen:
            plot_chromagram(st.session_state.gen_wav_data, st.session_state.gen_sr,
                            "Generated Music — Chromagram")

    # ── Analysis summary ─────────────────────────────────────────────────────
    analysis = st.session_state.audio_analysis
    if analysis and "error" not in analysis:
        st.markdown("---")
        st.markdown('<div class="section-header">🔬 Audio Analysis Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Key",        f"{analysis.get('key','?')} {analysis.get('mode','')}")
        c2.metric("Tempo",      f"{analysis.get('tempo', 0):.0f} BPM")
        c3.metric("Energy",     analysis.get("energy","?").split()[0].capitalize())
        c4.metric("Brightness", analysis.get("brightness","?").split()[0].capitalize())
        c5.metric("Duration",   f"{analysis.get('duration', 0):.1f} s")

        if st.session_state.conversion_prompt:
            st.markdown("**Conversion Prompt:**")
            st.markdown(
                f'<div class="prompt-box">{st.session_state.conversion_prompt}</div>',
                unsafe_allow_html=True,
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="🎵 AI Music Studio",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="studio-header">
      <h1 class="studio-title">🎵 AI Music Studio</h1>
      <p class="studio-sub">Generate original music · Import a tune · Transform it into any instrument</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎵 Generate Music", "🎸 Tune Converter", "📊 Visualizer"])

    with tab1:
        tab_generate()
    with tab2:
        tab_converter()
    with tab3:
        tab_visualizer()


if __name__ == "__main__":
    main()
