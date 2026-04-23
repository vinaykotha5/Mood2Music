FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0+cpu" \
    "torchaudio==2.6.0+cpu"

# NumPy 1.x for compatibility
RUN pip install --no-cache-dir "numpy<2"

# torchmetrics (required by audiocraft)
RUN pip install --no-cache-dir torchmetrics

# Install audiocraft dependencies manually (skip av which fails to build)
RUN pip install --no-cache-dir \
    einops \
    flashy \
    hydra-core \
    julius \
    num2words \
    scipy \
    sentencepiece \
    transformers \
    huggingface_hub \
    encodec \
    xformers || true

# Install audiocraft without dependencies, then install what we can
RUN pip install --no-cache-dir --no-deps audiocraft

# App dependencies
RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    librosa==0.11.0 \
    matplotlib \
    tensorboard \
    chromadb

COPY aaa.py app.py musicgen_wrapper.py \
     instrument_converter.py music_db.py ui_components.py \
     requirements.txt packages.txt README.md ./

RUN mkdir -p music_library/audio music_library/chroma

EXPOSE 7860

CMD ["streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
