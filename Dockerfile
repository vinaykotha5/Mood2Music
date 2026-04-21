FROM python:3.11-slim

WORKDIR /app

# System deps — only ffmpeg needed for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + audiocraft + app deps in a single pip call.
# Pinning torch version in the same call prevents pip from pulling CUDA torch.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.1.0+cpu" \
    "torchaudio==2.1.0+cpu" \
    audiocraft \
    streamlit==1.39.0 \
    librosa==0.11.0 \
    matplotlib \
    numpy \
    tensorboard \
    chromadb

# Copy ONLY the app source files (not everything)
COPY aaa.py app.py tensorflow_mock.py musicgen_wrapper.py \
     instrument_converter.py music_db.py ui_components.py \
     requirements.txt packages.txt README.md ./

RUN mkdir -p music_library/audio music_library/chroma

EXPOSE 7860

CMD ["streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
