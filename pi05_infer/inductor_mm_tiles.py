"""Extra small-M tile candidates for inductor's Triton ``mm`` template.

The denoise loop runs every expert GEMM at ``M = action_chunk`` (50 for the
turtle config), so the two weight-streaming GEMMs of each layer --
``down_proj`` (50x4096 @ 4096x1024) and ``o_proj`` (50x2048 @ 2048x1024) --
are pure bandwidth problems whose only parallelism is the tile grid.

Inductor's stock ``mm_kernel_configs`` list has no ``BLOCK_M = 16`` entry, and
``filtered_configs`` clamps every ``BLOCK_M`` down to ``next_power_of_2(50) =
64`` but never up from below.  The champion it lands on for both shapes is
``BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_stages=5, num_warps=4``.  With
``BLOCK_M=64`` and ``M=50`` the M dimension is a single tile, so the whole grid
is ``N / BLOCK_N = 32`` CTAs on a 110-SM card: the GEMM runs at roughly half
the bandwidth the same weight stream reaches elsewhere in the same layer.

The only ``BLOCK_M=32``/``BLOCK_K=128`` candidate the stock list does contain
carries ``num_stages=2``, which loses on pipeline depth rather than on CTA
count -- which is why the search never finds the good corner on its own.

This module appends a handful of deep-pipelined small-``BLOCK_M`` candidates
**for those two shapes only** and lets inductor's own autotuner benchmark them
against the stock list.  It picks ``BLOCK_M=16, BLOCK_N=64`` (or the equally
fast ``32/32``) -- either way 64 CTAs instead of 32.  MEASURED in the model on
RTX PRO 5000 Blackwell (sm_120, 110 SM), nsys 2026.1.2, 12 predicts::

    down_proj  15.06 -> 11.71 us/call  (-22.2%, 591 -> 760 GB/s)
    o_proj      8.47 ->  6.94 us/call  (-18.1%, 530 -> 647 GB/s)
    denoise stream busy  11.48 -> 10.55 ms/predict; kernel counts unchanged
    e2e paired A/B, SM clock held equal: -0.88 ms/predict

**Bit-exactness.**  ``BLOCK_K`` is the only tile parameter that changes the
order of the fp32 accumulation over K; ``BLOCK_M``/``BLOCK_N``/``num_warps``/
``num_stages`` only change which CTA owns which output element, not how that
element is computed.  Every candidate appended here therefore pins
``BLOCK_K = 128`` -- the value inductor's own champion already uses for both
shapes -- so whichever candidate wins produces bit-identical output.  All 15
configs in the sweep above returned byte-identical results.

**Scope.**  The patch replaces a name in ``torch._inductor.kernel.mm``, so it is
process-global: any *other* model compiled in the same process whose GEMM also
hits ``m <= 64`` with one of the listed ``(N, K)`` pairs gets the wider search
too.  That stays algebraically and numerically exact (``BLOCK_K`` is pinned), but
its kernel choice -- and therefore its output bits relative to an unpatched run --
can move.  Narrow ``RLINF_SMALL_M_MM_SHAPES`` or set ``RLINF_SMALL_M_MM=0`` if
that matters.

Kill switch: ``RLINF_SMALL_M_MM=0`` restores stock inductor behaviour.
Tuning: ``RLINF_SMALL_M_MM_CFGS="BM,BN,BK,stages,warps;..."`` and
``RLINF_SMALL_M_MM_SHAPES="N x K;..."``, ``RLINF_SMALL_M_MM_MAX_M``.
"""

import itertools
import os
from typing import Iterable, Optional

__all__ = ["install_small_m_mm_configs", "small_m_mm_enabled"]

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps). BLOCK_K MUST stay 128 for
# every entry -- see the bit-exactness note above.
_DEFAULT_CFGS = (
    (16, 64, 128, 5, 4),
    (16, 64, 128, 4, 4),
    (16, 64, 128, 3, 4),
    (16, 32, 128, 4, 2),
    (32, 32, 128, 5, 4),
)

# (N, K) of the GEMMs to widen the search for: the action expert's o_proj and
# down_proj. Everything else -- the 968-token PaliGemma prefix, the adaRMS
# projections, the action in/out projections -- keeps inductor's stock list, so
# its kernel selection (and hence its output bits) cannot move.
_DEFAULT_SHAPES = ((1024, 2048), (1024, 4096))

_REQUIRED_BLOCK_K = 128

_installed = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def small_m_mm_enabled() -> bool:
    """True when the extra small-M mm candidates should be registered."""
    return _env_int("RLINF_SMALL_M_MM", 1) != 0


def _parse_cfgs(raw: Optional[str]) -> tuple:
    if not raw:
        return _DEFAULT_CFGS
    cfgs = tuple(
        tuple(int(v) for v in entry.split(","))
        for entry in raw.split(";")
        if entry.strip()
    )
    for cfg in cfgs:
        assert len(cfg) == 5, (
            f"RLINF_SMALL_M_MM_CFGS entries must be 'BM,BN,BK,stages,warps', got {cfg}"
        )
        assert cfg[2] == _REQUIRED_BLOCK_K, (
            f"RLINF_SMALL_M_MM_CFGS: BLOCK_K must be {_REQUIRED_BLOCK_K} (it is the "
            f"only tile parameter that changes the fp32 reduction order over K, so "
            f"any other value silently breaks bit-exactness), got {cfg[2]} in {cfg}"
        )
    return cfgs


def _parse_shapes(raw: Optional[str]) -> tuple:
    if not raw:
        return _DEFAULT_SHAPES
    shapes = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        n, _, k = entry.partition("x")
        assert k, f"RLINF_SMALL_M_MM_SHAPES entries must be 'NxK', got {entry!r}"
        shapes.append((int(n), int(k)))
    return tuple(shapes)


def install_small_m_mm_configs() -> bool:
    """Widen inductor's mm autotune space for the denoise GEMM shapes.

    Idempotent; returns True when the patch is active. Fails soft (returns
    False) if this torch build does not expose the config hook, so a torch
    upgrade degrades to stock inductor rather than to a crash.
    """
    global _installed
    if _installed:
        return True
    if not small_m_mm_enabled():
        return False

    try:
        # Imported lazily: importing torch._inductor.kernel.mm at module scope
        # trips a circular import inside torch._inductor.lowering.
        import torch._inductor.lowering  # noqa: F401
        from torch._inductor.kernel import mm as mm_kernel
        from torch._inductor.kernel import mm_common

        original = mm_kernel.mm_configs
        filtered_configs = mm_common.filtered_configs
    except Exception:  # pragma: no cover - torch internals moved
        return False

    cfgs = _parse_cfgs(os.environ.get("RLINF_SMALL_M_MM_CFGS"))
    shapes = _parse_shapes(os.environ.get("RLINF_SMALL_M_MM_SHAPES"))
    max_m = _env_int("RLINF_SMALL_M_MM_MAX_M", 64)

    def _small_m_mm_configs(m, n, k, **kwargs) -> Iterable:
        base = original(m, n, k, **kwargs)
        try:
            applies = int(m) <= max_m and (int(n), int(k)) in shapes
        except (TypeError, ValueError):  # symbolic shape -> leave it alone
            return base
        if not applies:
            return base
        return itertools.chain(base, filtered_configs(m, n, k, configs=cfgs, **kwargs))

    mm_kernel.mm_configs = _small_m_mm_configs

    # The patch is invisible to inductor's FXGraphCache key, so without this a
    # run with the switch flipped would silently reuse the other arm's compiled
    # kernels -- which would make any A/B of this change meaningless.
    try:
        import torch.compiler.config as compiler_config

        tag = getattr(compiler_config, "cache_key_tag", "")
        compiler_config.cache_key_tag = f"{tag}+small_m_mm"
    except Exception:  # pragma: no cover - older torch without the tag
        pass

    _installed = True
    print(
        f"[pi05_infer] small-M mm tiles installed: shapes={list(shapes)} "
        f"max_m={max_m} extra_cfgs={list(cfgs)}"
    )
    return True
