"""
tensorflow_mock.py
──────────────────
Installs TensorBoard's own built-in `tensorflow_stub` as `sys.modules["tensorflow"]`
before audiocraft/flashy/torch.utils.tensorboard import it.

WHY THIS APPROACH
─────────────────
tensorboard.compat.tf is a LazyModule that tries:
  1. `import tensorflow`  → fails if not installed → falls back to tensorflow_stub
  2. `from tensorboard.compat import tensorflow_stub`

We short-circuit step 1 by pre-installing tensorflow_stub in sys.modules so that
`import tensorflow` succeeds and returns the proper stub.

tensorflow_stub is TensorBoard's own production-quality TF mock. It already has:
  - tf.io / tf.io.gfile (GFile, join, get_filesystem)
  - tf.compat.as_bytes
  - tf.errors, tf.dtypes, tf.summary, ...

We just add the handful of attrs that torch.utils.tensorboard and flashy also
check for (tf.Tensor, tf.Variable) so those never raise AttributeError.

Unlike a home-grown mock, tensorflow_stub has a real __file__ path so Python's
inspect.getfile() never throws "is a built-in module" errors.
"""

import sys
import importlib.machinery


def _install_tf_stub() -> None:
    """Idempotent: installs the TF stub exactly once."""

    # If real TF is there and works — leave it alone
    if "tensorflow" in sys.modules:
        existing = sys.modules["tensorflow"]
        # Our stub already installed
        if getattr(existing, "_IS_TB_STUB", False):
            return
        # Real TF that already has Variable/Tensor — fine
        if hasattr(existing, "Variable") and hasattr(existing, "Tensor"):
            return

    # ── Load TensorBoard's own TF stub ───────────────────────────────────────
    try:
        from tensorboard.compat import tensorflow_stub as _stub
    except ImportError:
        # TensorBoard not installed either — create a minimal emergency mock
        _stub = None

    if _stub is not None:
        # Mark it so we can detect it on re-runs
        _stub._IS_TB_STUB = True

        # Add tensorflow.Tensor if missing (isinstance checks in torch TB utils)
        if not hasattr(_stub, "Tensor"):
            class _FakeTensor:
                """Fake TensorFlow Tensor class for isinstance checks."""
                def __init__(self, *args, **kwargs):
                    pass
                def __repr__(self):
                    return "<FakeTensor>"
            _stub.Tensor = _FakeTensor

        # Add tensorflow.Variable if missing (accessed during model.generate)
        if not hasattr(_stub, "Variable"):
            class _FakeVariable:
                """Fake TensorFlow Variable class for isinstance checks."""
                def __init__(self, *args, **kwargs):
                    pass
                def __repr__(self):
                    return "<FakeVariable>"
            _stub.Variable = _FakeVariable

        # Patch tf.io.gfile with any missing methods that torch TB needs
        try:
            gfile = _stub.io.gfile
            if not hasattr(gfile, "join"):
                gfile.join = lambda *parts: "/".join(str(p) for p in parts)
            if not hasattr(gfile, "GFile"):
                gfile.GFile = open
            if not hasattr(gfile, "get_filesystem"):
                import types as _types
                gfile.get_filesystem = lambda path: _types.SimpleNamespace(
                    join=lambda a, b: str(a) + "/" + str(b)
                )
        except Exception:
            pass

        # Give it a proper __spec__ so importlib.util.find_spec won't break
        if not hasattr(_stub, "__spec__") or _stub.__spec__ is None:
            spec = importlib.machinery.ModuleSpec(
                "tensorflow",
                loader=None,
                origin=getattr(_stub, "__file__", None),
            )
            spec.submodule_search_locations = []
            _stub.__spec__ = spec

        # Register as sys.modules["tensorflow"] so `import tensorflow` returns it
        sys.modules["tensorflow"] = _stub

        # Also register submodules so `from tensorflow import io` works
        for attr in ("io", "compat", "errors", "dtypes", "summary"):
            sub = getattr(_stub, attr, None)
            if sub is not None and attr not in sys.modules.get("tensorflow." + attr, {}):
                sys.modules[f"tensorflow.{attr}"] = sub

    else:
        # ── Emergency fallback: plain module with __getattr__ ────────────────
        import types

        class _EmergencyMock(types.ModuleType):
            _IS_TB_STUB = True

            class _AnyStub:
                def __call__(self, *a, **kw): return self
                def __getattr__(self, n): return type(self)()
                def __bool__(self): return True
                def __enter__(self): return self
                def __exit__(self, *a): return False

            def __getattr__(self, name):
                stub = _EmergencyMock._AnyStub()
                object.__setattr__(self, name, stub)
                return stub

        mock = _EmergencyMock("tensorflow")
        # MUST have __file__ so inspect.getfile doesn't raise TypeError
        mock.__file__ = __file__
        mock._IS_TB_STUB = True
        # MUST have __spec__ so importlib.util.find_spec() doesn't crash
        mock.__spec__ = importlib.machinery.ModuleSpec(
            "tensorflow", loader=None, origin=__file__
        )
        sys.modules["tensorflow"] = mock

    # ── Patch tensorboard's lazy tf proxy so it never tries to re-import ─────
    try:
        import tensorboard.compat as _tbc
        tf_value = sys.modules["tensorflow"]
        # Overwrite the LazyModule's internal cache so it returns our stub directly
        _tbc.__dict__["tf"] = tf_value
    except Exception:
        pass


# Run immediately on import — MUST happen before any audiocraft/flashy import
_install_tf_stub()
