FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for audio, git for audiocraft install
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only PyTorch FIRST (prevents pip from pulling the 2.5GB CUDA build)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install audiocraft using direct git URL (not PEP 508 @ syntax)
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/audiocraft.git

# 3. Install remaining dependencies
RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    librosa==0.11.0 \
    matplotlib \
    numpy \
    tensorboard==2.20.0 \
    chromadb \
    "packaging>=23.1,<25"

# Copy app code
COPY . .

# Create library directory
RUN mkdir -p music_library/audio music_library/chroma

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
