FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Write a constraints file so pip won't upgrade/replace torch with CUDA
#    then install audiocraft WITH all its deps (no --no-deps)
RUN echo "torch==2.1.0+cpu" > /tmp/constraints.txt && \
    echo "torchaudio==2.1.0+cpu" >> /tmp/constraints.txt && \
    pip install --no-cache-dir \
        --constraint /tmp/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        audiocraft

# 3. App dependencies
RUN pip install --no-cache-dir \
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
