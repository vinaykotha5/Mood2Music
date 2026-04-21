"""
Stub for xformers.ops — provides the memory_efficient_attention interface
that audiocraft checks for, but raises NotImplementedError so audiocraft
falls back to standard PyTorch multi-head attention on CPU.
"""


class _MemoryEfficientAttentionOp:
    """Placeholder op — audiocraft checks `if op is not None`."""
    pass


def memory_efficient_attention(query, key, value, attn_bias=None, op=None, scale=None):
    """
    Stub that raises so audiocraft's try/except catches it
    and falls back to torch.nn.functional.scaled_dot_product_attention.
    """
    raise NotImplementedError("xformers not available (CPU-only deployment)")


def memory_efficient_attention_forward(*args, **kwargs):
    raise NotImplementedError("xformers not available (CPU-only deployment)")


class LowerTriangularMask:
    pass


# Some audiocraft code checks for fmha
class fmha:
    memory_efficient_attention = staticmethod(memory_efficient_attention)
    attn_bias = type('attn_bias', (), {'LowerTriangularMask': LowerTriangularMask})
