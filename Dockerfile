FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.11
RUN conda create -n audiocraft python=3.11 -y

# Activate environment and install packages
SHELL ["conda", "run", "-n", "audiocraft", "/bin/bash", "-c"]

# Install PyAV from conda-forge (pre-built binary)
RUN conda install -c conda-forge av -y

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    torchaudio==2.6.0+cpu

# NumPy 1.x for compatibility
RUN pip install --no-cache-dir "numpy<2"

# torchmetrics
RUN pip install --no-cache-dir torchmetrics

# Install audiocraft
RUN pip install --no-cache-dir audiocraft

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

# Run with conda environment activated
CMD ["conda", "run", "--no-capture-output", "-n", "audiocraft", "streamlit", "run", "aaa.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
