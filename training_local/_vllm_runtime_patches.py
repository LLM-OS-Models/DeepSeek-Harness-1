"""Runtime monkey-patches for vLLM 0.25 that activate on import.

Patches:
  - MHC ops `forward_cuda` hardcoded to tilelang. tilelang 0.1.9 doesn't
    work on Python 3.12 (tvm_ffi `__dict__` not writable). When tilelang
    isn't importable, delegate `forward_cuda` to `forward_hip`, which has
    a proper `HAS_TILELANG_MHC`-gated fallback to `forward_native`.

Import this module before constructing any vLLM `LLM(...)`:
    from training_local._vllm_runtime_patches import apply_all
    apply_all()
"""
from __future__ import annotations


def patch_mhc_forward_cuda_fallback() -> bool:
    """Return True if patches were applied (or already applied)."""
    try:
        from vllm.model_executor.layers import mhc as _mhc
    except ImportError:
        return False

    if getattr(_mhc, "_harness1_mhc_patched", False):
        return True

    # If tilelang is healthy, no patch needed.
    if _mhc.HAS_TILELANG_MHC:
        return True

    targets = ("MHCPreOp", "MHCPostOp", "MHCFusedPostPreOp", "HCHeadOp")
    patched = 0
    for cls_name in targets:
        cls = getattr(_mhc, cls_name, None)
        if cls is None:
            continue
        # forward_hip has the proper aiter/tilelang/native dispatch.
        # On CUDA + (HAS_TILELANG_MHC=False), it falls through to
        # forward_native — same semantics we want for forward_cuda.
        hip_fn = cls.__dict__.get("forward_hip")
        cuda_fn = cls.__dict__.get("forward_cuda")
        if hip_fn is None or cuda_fn is None:
            continue
        cls.forward_cuda = hip_fn
        patched += 1

    _mhc._harness1_mhc_patched = True
    return patched == len(targets)


def apply_all() -> None:
    patch_mhc_forward_cuda_fallback()
