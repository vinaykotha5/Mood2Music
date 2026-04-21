FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch first — prevents pip from resolving to the 2.5GB CUDA build
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. All other deps
RUN pip install --no-cache-dir \
    audiocraft \
    streamlit==1.39.0 \
    librosa==0.11.0 \
    matplotlib \
    numpy \
    tensorboard==2.20.0 \
    chromadb \
    "packaging>=23.1,<25"

# Copy app
COPY . .

RUN mkdir -p music_library/audio music_library/chroma

EXPOSE 7860

CMD ["streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
