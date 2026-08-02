#!/usr/bin/env python3
"""Apply post-install patches to vLLM that can't be done via requirements files.

Patches:
  1. NamespaceTool import alias in vllm/tool_parsers/utils.py.
     vLLM 0.25 references `NamespaceTool` from openai.types.responses, which
     only exists in openai >=2.52. Stable openai wheels don't have it. The
     patch wraps the import in try/except and aliases to FunctionTool on
     ImportError.
  2. tilelang stub library replacement.
     tilelang 0.1.9 ships its own libcudart_stub.so / libcuda_stub.so /
     libnvrtc_stub.so which are incomplete (e.g. missing `cudaDeviceReset`).
     When vLLM's mhc_kernels imports tilelang in a worker process where
     CUDA is already initialized, ctypes lookup of common CUDA symbols
     fails with AttributeError → ImportError → "tilelang is required for
     mhc but is not installed" runtime error.
     The patch replaces the three stubs with symlinks to the real system
     CUDA libraries.

Idempotent: detects existing patches and skips. Run after every vllm
reinstall:
    python -m training_local.patch_vllm
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PATCH_MARKER = "# harness-1 patch: NamespaceTool alias"


def patch_namespacetool(utils_path: Path) -> bool:
    """Return True if the file was modified."""
    text = utils_path.read_text()
    if PATCH_MARKER in text:
        return False

    target = "from openai.types.responses import (\n    FunctionTool,\n    NamespaceTool,\n    ToolChoiceFunction,\n)"
    replacement = (
        "from openai.types.responses import (\n"
        "    FunctionTool,\n"
        "    ToolChoiceFunction,\n"
        ")\n"
        f"{PATCH_MARKER}\n"
        "try:\n"
        "    from openai.types.responses import NamespaceTool\n"
        "except ImportError:\n"
        "    NamespaceTool = FunctionTool\n"
    )

    if target not in text:
        print(f"WARN: target block not found in {utils_path}", file=sys.stderr)
        print("vLLM may have fixed upstream, or file layout changed.", file=sys.stderr)
        return False

    utils_path.write_text(text.replace(target, replacement))
    return True


def _find_real_lib(name: str) -> Path | None:
    """Locate a real CUDA library via ldconfig."""
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if f"{name} " in line or line.strip().endswith(name):
            arrow = line.rfind("=>")
            if arrow != -1:
                path = line[arrow + 2:].strip().split()[0]
                if path and Path(path).exists():
                    return Path(path)
    return None


MHC_PATCH_MARKER = "# harness-1 patch: mhc forward_cuda fallback"


def patch_mhc_forward_cuda(mhc_path: Path) -> int:
    """Insert HAS_TILELANG_MHC gate at top of each forward_cuda body.

    vLLM 0.25's MHC ops hardcode tilelang in forward_cuda. tilelang 0.1.9
    is broken on Python 3.12 (tvm_ffi __dict__ issue), so when tilelang
    isn't available we delegate forward_cuda to the torch/triton fallback
    (same logic as forward_hip's `else` branch).

    Returns count of forward_cuda methods patched (0 if already patched).
    """
    text = mhc_path.read_text()
    if MHC_PATCH_MARKER in text:
        return 0

    pre_orig = """    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip("""

    pre_new = """    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not HAS_TILELANG_MHC:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip("""

    post_orig = """    ) -> torch.Tensor:
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )"""

    post_new = """    ) -> torch.Tensor:
        if not HAS_TILELANG_MHC:
            return self.forward_native(
                x, residual, post_layer_mix, comb_res_mix
            )
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )"""

    head_orig = """        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)"""

    head_new = """        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        if not HAS_TILELANG_MHC:
            num_tokens = hs_flat.shape[0]
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )
            return out.view(*outer_shape, hidden_size)
        out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)"""

    fused_orig = """    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )"""

    fused_new = """    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not HAS_TILELANG_MHC:
            return self.forward_native(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )"""

    pairs = [
        ("MHCPreOp.forward_cuda", pre_orig, pre_new),
        ("MHCPostOp.forward_cuda", post_orig, post_new),
        ("HCHeadOp.forward_cuda", head_orig, head_new),
        ("MHCFusedPostPreOp.forward_cuda", fused_orig, fused_new),
    ]

    patches_applied = 0
    new_text = text
    new_text += f"\n{MHC_PATCH_MARKER}\n"
    for name, old, new in pairs:
        if old not in new_text:
            print(
                f"WARN: {name} target block not found in {mhc_path}",
                file=sys.stderr,
            )
            continue
        new_text = new_text.replace(old, new, 1)
        patches_applied += 1

    if patches_applied > 0:
        mhc_path.write_text(new_text)
    return patches_applied


def patch_tilelang_stubs() -> int:
    """Replace tilelang stubs with symlinks to real CUDA libs.

    Returns count of stubs replaced.
    """
    try:
        import tilelang
    except Exception:
        # tilelang may be installed but broken on Py 3.12 (tvm_ffi issue).
        # Locate via site-packages instead.
        import sys
        tilelang_dir = None
        for p in sys.path:
            candidate = Path(p) / "tilelang"
            if candidate.is_dir():
                tilelang_dir = candidate
                break
        if tilelang_dir is None:
            print("tilelang not installed — skipping stub patch", file=sys.stderr)
            return 0
        tilelang_lib_dir = tilelang_dir / "lib"
    else:
        tilelang_lib_dir = Path(tilelang.__file__).parent / "lib"

    if not tilelang_lib_dir.exists():
        print(f"WARN: {tilelang_lib_dir} does not exist", file=sys.stderr)
        return 0

    candidates = [
        ("libcudart_stub.so", "libcudart.so.12"),
        ("libcuda_stub.so", "libcuda.so.1"),
        ("libnvrtc_stub.so", "libnvrtc.so.12"),
    ]
    replaced = 0
    for stub_name, real_name in candidates:
        stub_path = tilelang_lib_dir / stub_name
        if not stub_path.exists():
            continue
        if stub_path.is_symlink():
            continue
        real_path = _find_real_lib(real_name)
        if real_path is None:
            print(f"WARN: could not find real {real_name} via ldconfig", file=sys.stderr)
            continue
        backup = stub_path.with_suffix(stub_path.suffix + ".bak")
        if not backup.exists():
            stub_path.rename(backup)
        else:
            stub_path.unlink()
        stub_path.symlink_to(real_path)
        print(f"  {stub_name} -> {real_path}")
        replaced += 1
    return replaced


CUTE_PATCH_MARKER = "# harness-1 patch: back-compat for vLLM 0.25 fmha code"


def patch_cute_core_thrmma() -> bool:
    """Add `core.ThrMma = ThrMma` back-compat alias to cutlass.cute.__init__.

    vLLM 0.25's bundled fmha_sm100 / vllm_flash_attn files reference
    `cute.core.ThrMma` in function annotations. cutlass-dsl 4.6.x exposes
    ThrMma only at `cute.ThrMma`. Because those files don't use
    `from __future__ import annotations`, Python evaluates the annotation
    eagerly at import time and dies with:
        AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'

    The fix appends a single line to cutlass/cute/__init__.py that sets
    `core.ThrMma = ThrMma` after the symbol is imported.

    Returns True if the file was modified.
    """
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False

    cutlass_dir = Path(cutlass.__file__).parent
    cute_init_path = cutlass_dir / "cute" / "__init__.py"
    if not cute_init_path.exists():
        return False

    text = cute_init_path.read_text()
    if CUTE_PATCH_MARKER in text:
        return False

    # The file ends with `]` (closing the __all__ list). Append the alias.
    new_text = text.rstrip() + '\n\n' + CUTE_PATCH_MARKER + '\n' \
        'core.ThrMma = ThrMma  # type: ignore[attr-defined]\n'
    cute_init_path.write_text(new_text)
    return True


FMAX_PATCH_MARKER = "# harness-1 patch: force nvvm.fmax new API (2 positional args)"


def patch_fmax_cuda129_branch() -> int:
    """Replace `CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9` with `False`.

    vLLM 0.25's bundled `vllm_flash_attn/cute/utils.py` and
    `third_party/fmha_sm100/cute/src/common/utils.py` branch on CUDA 12.9 to
    pick between "old" and "new" nvvm.fmax API. The "old" path passes
    `T.f32()` as a leading positional argument, but cutlass-dsl 4.6.x's
    `nvvm.fmax` only accepts 2 positional args (new API). On systems where
    the driver reports CUDA 12.9 (e.g. driver 570.86.10) but cutlass-dsl
    uses the new MLIR op signature, the if-branch fails with:
        TypeError: fmax() takes 2 positional arguments but 3 positional
        arguments (and 3 keyword-only arguments) were given

    We force-skip the if-branch so the new-API else-branch always runs.

    Returns count of files modified.
    """
    try:
        import vllm
    except ImportError:
        return 0

    vllm_dir = Path(vllm.__file__).parent
    targets = [
        vllm_dir / "vllm_flash_attn" / "cute" / "utils.py",
        vllm_dir / "third_party" / "fmha_sm100" / "cute" / "src" / "common" / "utils.py",
    ]
    old_cond = "if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:"
    new_cond = "if False:  # " + FMAX_PATCH_MARKER + " (forced new API)"
    patched = 0
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text()
        if FMAX_PATCH_MARKER in text:
            continue
        if old_cond not in text:
            continue
        path.write_text(text.replace(old_cond, new_cond, 1))
        patched += 1
    return patched


def main() -> int:
    try:
        import vllm
    except ImportError:
        print("vllm not installed — nothing to patch", file=sys.stderr)
        return 1

    vllm_dir = Path(vllm.__file__).parent
    utils_path = vllm_dir / "tool_parsers" / "utils.py"
    if not utils_path.exists():
        print(f"WARN: {utils_path} does not exist", file=sys.stderr)
        return 1

    changed = patch_namespacetool(utils_path)
    if changed:
        print(f"PATCHED {utils_path} (NamespaceTool alias)")
    else:
        print(f"OK {utils_path} (already patched or upstream-fixed)")

    mhc_path = vllm_dir / "model_executor" / "layers" / "mhc.py"
    if mhc_path.exists():
        mhc_count = patch_mhc_forward_cuda(mhc_path)
        if mhc_count > 0:
            print(f"PATCHED {mhc_path} ({mhc_count} forward_cuda methods)")
        else:
            print(f"OK {mhc_path} (already patched)")
    else:
        print(f"WARN: {mhc_path} does not exist", file=sys.stderr)

    stub_count = patch_tilelang_stubs()
    if stub_count > 0:
        print(f"PATCHED tilelang stubs ({stub_count} replaced)")
    else:
        print("OK tilelang stubs (already symlinked or tilelang absent)")

    cute_changed = patch_cute_core_thrmma()
    if cute_changed:
        cute_path = Path(__import__("cutlass").__file__).parent / "cute" / "__init__.py"
        print(f"PATCHED {cute_path} (cute.core.ThrMma back-compat)")
    else:
        print("OK cutlass.cute (already patched or cutlass absent)")

    fmax_count = patch_fmax_cuda129_branch()
    if fmax_count > 0:
        print(f"PATCHED nvvm.fmax CUDA-12.9 branch ({fmax_count} files)")
    else:
        print("OK nvvm.fmax branch (already patched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
