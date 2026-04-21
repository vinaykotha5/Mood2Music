FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. CPU-only PyTorch — must come before audiocraft to prevent CUDA resolution
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. audiocraft (will reuse torch already installed above)
RUN pip install --no-cache-dir --no-deps audiocraft
RUN pip install --no-cache-dir \
    "einops>=0.6.1" \
    "flashy>=0.0.1" \
    "hydra-core>=1.1" \
    "num2words" \
    "scipy>=1.9.0" \
    "sentencepiece" \
    "spacy>=3.0.0" \
    "transformers>=4.31.0" \
    "xformers" || true

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
