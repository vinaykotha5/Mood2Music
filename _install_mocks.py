"""
_install_mocks.py
─────────────────
Pre-installs mock modules into sys.modules for ALL heavy packages
that audiocraft imports at the module level but doesn't need for
MusicGen CPU inference.

MUST be imported BEFORE `from audiocraft.models import MusicGen`.

This eliminates the need to install these packages (saving ~500MB+):
  - spacy (~120MB) — used for TextConditioner, not needed by MusicGen
  - xformers (~200MB) — GPU attention, MusicGen uses PyTorch attention on CPU
  - demucs (~50MB) — source separation, not used by MusicGen
  - soundfile (~5MB) — audiocraft uses av for audio I/O instead
  - audioseal (~30MB) — watermarking, not used by MusicGen
  - torchdiffeq (~10MB) — ODE solvers for flow matching, not MusicGen
  - laion_clap (~50MB) — CLAP conditioning, not used by MusicGen
  - dora (~5MB) — training framework, not used at inference
"""

import sys
import types
import importlib.machinery


class _StubClass(type):
    """Metaclass for creating stub classes that can be used in isinstance checks."""
    def __instancecheck__(cls, instance):
        return False
    
    def __subclasscheck__(cls, subclass):
        return False


class _StubObj(metaclass=_StubClass):
    """Universal stub that absorbs any attribute access or call."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *a, **kw):
        return _StubObj()
    
    def __getattr__(self, name):
        # Return a class for common type-like attributes
        if name in ('Tensor', 'Variable', 'Module', 'Parameter', 'Model', 
                    'Layer', 'Sequential', 'Embedding', 'Linear'):
            return type(name, (_StubObj,), {})
        return _StubObj()
    
    def __bool__(self):
        return False
    
    def __iter__(self):
        return iter([])
    
    def __len__(self):
        return 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, *a):
        return False
    
    def __repr__(self):
        return "<MockStub>"


def _make_mock_module(name: str) -> types.ModuleType:
    """Create a mock module that returns _StubObj for any attribute."""
    mod = types.ModuleType(name)
    mod.__file__ = __file__
    mod.__path__ = []
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    mod.__all__ = []
    mod.__version__ = "0.0.0"
    # Proper __spec__ so importlib.util.find_spec() doesn't crash
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=__file__)

    def _getattr(attr_name):
        # Return proper classes for common type names
        if attr_name in ('Tensor', 'Variable', 'Module', 'Parameter', 'Model',
                         'Layer', 'Sequential', 'Embedding', 'Linear', 'Doc',
                         'Span', 'Token', 'Language', 'Vocab'):
            return type(attr_name, (_StubObj,), {'__module__': name})
        return _StubObj()

    mod.__getattr__ = _getattr
    return mod


# ─── All packages to mock ────────────────────────────────────────────────────
# Each entry is a top-level package name.
# We also register common submodules that audiocraft imports explicitly.

_MOCKS = {
    # spacy — conditioners.py: `import spacy` at module level
    "spacy": [
        "spacy", "spacy.tokens", "spacy.tokens.doc", "spacy.tokens.span",
        "spacy.language", "spacy.lang", "spacy.lang.en",
        "spacy.cli", "spacy.util",
    ],
    # xformers — transformer.py: `from xformers import ops` / profiler / checkpoint
    "xformers": [
        "xformers", "xformers.ops",
        "xformers.profiler", "xformers.checkpoint_fairinternal",
    ],
    # demucs — conditioners.py: `from demucs.apply/audio import ...`
    "demucs": [
        "demucs", "demucs.pretrained", "demucs.apply",
        "demucs.audio", "demucs.hdemucs", "demucs.states",
    ],
    # soundfile — data/audio.py: `import soundfile`
    "soundfile": [
        "soundfile",
    ],
    # audioseal — models/watermark.py
    "audioseal": [
        "audioseal", "audioseal.builder",
    ],
    # torchdiffeq — models/flow_matching.py
    "torchdiffeq": [
        "torchdiffeq",
    ],
    # laion_clap — conditioners.py (inside try/except, but just in case)
    "laion_clap": [
        "laion_clap", "laion_clap.clap_module",
    ],
    # dora — flashy/distrib.py & flashy/logging.py: `from dora.distrib import ...`
    "dora": [
        "dora", "dora.distrib", "dora.log",
    ],
}


def install_all_mocks():
    """Register all mock modules. Idempotent — safe to call multiple times."""
    for _pkg, submodules in _MOCKS.items():
        for mod_name in submodules:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = _make_mock_module(mod_name)


# Run immediately on import
install_all_mocks()
