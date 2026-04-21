FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.1.0+cpu" \
    "torchaudio==2.1.0+cpu"

# audiocraft without its heavy optional deps (spacy, xformers)
# Install only deps needed for MusicGen inference
RUN pip install --no-cache-dir --no-deps audiocraft && \
    pip install --no-cache-dir \
    av julius einops flashy \
    "transformers>=4.31.0" \
    sentencepiece scipy \
    num2words hydra-core \
    huggingface_hub encodec

# App deps (no tensorboard, no chromadb — saves ~400MB)
RUN pip install --no-cache-dir \
    streamlit==1.39.0 \
    librosa==0.11.0 \
    matplotlib \
    numpy

COPY aaa.py app.py tensorflow_mock.py musicgen_wrapper.py \
     instrument_converter.py music_db.py ui_components.py \
     requirements.txt packages.txt README.md ./
COPY xformers/ ./xformers/

RUN mkdir -p music_library/audio music_library/chroma

EXPOSE 7860

CMD ["streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
