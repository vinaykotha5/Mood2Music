"""
Lightweight xformers stub for CPU-only deployments.
Audiocraft imports `from xformers import ops` but only uses GPU memory-efficient
attention when CUDA is available. This stub provides the interface so the import
succeeds, while audiocraft falls back to standard PyTorch attention on CPU.
"""
