"""
Quick test to verify all imports work correctly
"""
import sys

print("Testing imports...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import torchaudio
    print(f"✓ torchaudio {torchaudio.__version__}")
except Exception as e:
    print(f"✗ torchaudio: {e}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import torchmetrics
    print(f"✓ torchmetrics {torchmetrics.__version__}")
except Exception as e:
    print(f"✗ torchmetrics: {e}")

try:
    from audiocraft.models import MusicGen
    print(f"✓ audiocraft.models.MusicGen")
except Exception as e:
    print(f"✗ audiocraft: {e}")

try:
    import streamlit
    print(f"✓ streamlit {streamlit.__version__}")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import librosa
    print(f"✓ librosa {librosa.__version__}")
except Exception as e:
    print(f"✗ librosa: {e}")

print("\nAll critical imports successful!")
