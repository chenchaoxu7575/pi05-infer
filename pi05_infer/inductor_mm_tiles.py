"""Extra small-M tile candidates for inductor's Triton ``mm``/``bmm`` templates.

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

--------------------------------------------------------------------------
The attention BMMs (``install_small_m_bmm_configs``, ``RLINF_SMALL_M_BMM``)
--------------------------------------------------------------------------

``torch._inductor.kernel.bmm`` binds ``mm_configs`` at import time
(``from .mm_common import mm_configs``), so replacing
``torch._inductor.kernel.mm.mm_configs`` above never reached the two batched
attention GEMMs of the denoise loop.  They are patched separately here:

    P.V    ``bmm(8x50x1018, 8x1018x256)``  -> ``triton_tem_fused_bmm_7``
    Q.K^T  ``bmm(8x50x256 , 8x256x1018)``  -> ``triton_tem_fused_bmm_5``

(8 query heads, ``action_chunk`` 50 rows, 968 prefix + 50 suffix = 1018 KV,
head_dim 256; the single KV head is broadcast, so the second operand's batch
stride is 0 and each byte of K/V is read once from DRAM.)

Only ``P.V`` gets extra candidates, and the winning candidate is *not* the one
the occupancy analysis predicted.  Measured in isolation on the real shapes,
SM clock locked at 2092 MHz, kernel time from the CUDA profiler over 50 graph
replays (``claude_mem/pi05_rollout_forward/kernel_fusion/probe_bmm_tile.py``)::

    P.V   stock champion  BM64 BN32 BK128 s5 w4   64 CTA   7.79 us
          appended        BM32 BN64 BK128 s4 w4   64 CTA   5.97 us  (-23.3%)
          BM16 (256 CTA)  BM16 BN32/64 BK128      256 CTA  8.0-8.3 us  (WORSE)

The grid is 64 CTAs either way -- ``ceil(50/64)*ceil(256/32) = 8`` versus
``ceil(50/32)*ceil(256/64) = 8``, times 8 batch.  Quadrupling the CTA count
with ``BLOCK_M=16``, which is what the ncu occupancy profile pointed at (46 of
110 SMs never execute a cycle), makes this kernel 30% *slower*.  The win comes
from the tile aspect ratio and one less pipeline buffer (96 KB -> 72 KB of
shared memory), not from filling the machine.  Inductor's own autotune in the
model agrees: 0.0102 ms for the stock champion, 0.0082 ms for this entry.

``Q.K^T`` gets **nothing**: it was swept over 28 configs and the best one found
(``BM64 BN128 BK32 s4 w4``, 4.42 us) beats the stock champion (4.49 us) by 1.6%,
which is below the noise floor of inductor's own autotune benchmarking.  That
benchmarking cannot separate the top of this shape's field at all -- in a
production compile its nine best choices all land within 0.0056-0.0062 ms, and
they carry ``BLOCK_K`` 32, 64 *and* 128.  Which one wins therefore moves from
process to process on a cold cache, and with it the fp32 reduction split: this
shape is not bit-stable across fresh autotunes even with no patch at all.  That
is a pre-existing hazard (mitigated in practice by ``PersistentCache`` pinning
the winner per cache dir), and a reason not to add candidates here.

**Bit-exactness for the BMMs.**  Same rule as above, with one correction that
matters: the pinned ``BLOCK_K`` is a property of the *shape*, not a constant --
it is whatever the stock champion for that shape uses, because that is the
accumulation order the unpatched build produces.  ``_DEFAULT_BMM_SHAPES``
therefore carries the required ``BLOCK_K`` per shape and every candidate for
that shape is asserted against it, rather than against a module-wide 128.

Kill switch: ``RLINF_SMALL_M_BMM=0``.
Tuning: ``RLINF_SMALL_M_BMM_SHAPES="NxK@BK;..."``,
``RLINF_SMALL_M_BMM_CFGS="NxK:BM,BN,BK,stages,warps|...;..."``,
``RLINF_SMALL_M_BMM_MAX_M``.
"""

import itertools
import os
from typing import Iterable, Optional

__all__ = [
    "install_small_m_mm_configs",
    "small_m_mm_enabled",
    "install_small_m_bmm_configs",
    "small_m_bmm_enabled",
]

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps). BLOCK_K MUST stay 128 for
# every entry -- see the bit-exactness note above.
#
# Deliberately NOT extended with BLOCK_N=32 / num_warps=4 entries. The theory
# that "(16,32,128,*,2) only loses because it is stuck at 2 warps, so give it 4
# and reach 128 CTAs" is wrong twice over:
#   * filtered_configs clamps num_warps to BLOCK_M*BLOCK_N//256, which is 2 for
#     a 16x32 tile, so (16,32,128,4,4) is silently rewritten to the entry that
#     is already in this list;
#   * forcing 4 warps past that clamp changes nothing anyway -- measured 14.479
#     us at 2 warps and 14.479 us at 4 (down_proj, 2092 MHz, kernel time).
# The 128-CTA corner is reachable today and is simply slower: down_proj 14.48 us
# at BN=32 versus 13.90 us at BN=64, o_proj 9.18 versus 8.56.
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

# ---- attention bmm ----------------------------------------------------------
# (N, K) -> BLOCK_K that the *stock* champion for that shape already uses. Every
# appended candidate for the shape must carry exactly this BLOCK_K: it is the
# only tile parameter that changes how the fp32 accumulation over K is split,
# so a different value would silently stop being bit-exact w.r.t. an unpatched
# build. Note the two shapes disagree -- P.V pins 128, Q.K^T pins 32.
_DEFAULT_BMM_SHAPES = {
    (256, 1018): 128,  # P.V   bmm(8x50x1018, 8x1018x256)
}

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps) per (N, K).
_DEFAULT_BMM_CFGS = {
    # P.V. One entry, not a family: the isolated sweep at a locked 2092 MHz put
    # num_stages=4 at 5.97 us, 3 at 6.07 and 5 at 6.31, while inductor's own
    # autotune timed all three at 0.0082 ms and would have picked between them
    # by benchmark noise. Widen with RLINF_SMALL_M_BMM_CFGS if another card
    # wants a different corner.
    (256, 1018): ((32, 64, 128, 4, 4),),
}

_installed = False
_bmm_installed = False


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
    _bump_cache_key_tag("small_m_mm")

    _installed = True
    print(
        f"[pi05_infer] small-M mm tiles installed: shapes={list(shapes)} "
        f"max_m={max_m} extra_cfgs={list(cfgs)}"
    )
    return True


def _bump_cache_key_tag(tag: str) -> None:
    """Make the monkeypatch visible to inductor's FXGraphCache key.

    Without this a run with the switch flipped would silently reuse the other
    arm's compiled kernels -- which would make any A/B of this change
    meaningless.
    """
    try:
        import torch.compiler.config as compiler_config

        prev = getattr(compiler_config, "cache_key_tag", "")
        compiler_config.cache_key_tag = f"{prev}+{tag}"
    except Exception:  # pragma: no cover - older torch without the tag
        pass


def small_m_bmm_enabled() -> bool:
    """True when the extra tile candidates for the attention BMMs should load."""
    return _env_int("RLINF_SMALL_M_BMM", 1) != 0


def _parse_bmm_shapes(raw: Optional[str]) -> dict:
    """``"NxK@BK;..."`` -> ``{(N, K): required_block_k}``."""
    if not raw:
        return dict(_DEFAULT_BMM_SHAPES)
    shapes = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        nk, _, bk = entry.partition("@")
        n, _, k = nk.partition("x")
        assert bk and k, (
            f"RLINF_SMALL_M_BMM_SHAPES entries must be 'NxK@BLOCK_K' -- the "
            f"BLOCK_K is the accumulation-order pin and is not optional, got "
            f"{entry!r}"
        )
        shapes[(int(n), int(k))] = int(bk)
    return shapes


def _parse_bmm_cfgs(raw: Optional[str], shapes: dict) -> dict:
    """``"NxK:BM,BN,BK,s,w|...;..."`` -> ``{(N, K): ((BM, BN, BK, s, w), ...)}``."""
    cfgs = dict(_DEFAULT_BMM_CFGS) if not raw else {}
    if raw:
        for entry in raw.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            nk, _, rest = entry.partition(":")
            n, _, k = nk.partition("x")
            assert rest and k, (
                f"RLINF_SMALL_M_BMM_CFGS entries must be "
                f"'NxK:BM,BN,BK,stages,warps|...', got {entry!r}"
            )
            cfgs[(int(n), int(k))] = tuple(
                tuple(int(v) for v in c.split(",")) for c in rest.split("|") if c.strip()
            )
    for shape, entries in cfgs.items():
        required_bk = shapes.get(shape)
        for cfg in entries:
            assert len(cfg) == 5, (
                f"bmm tile entries must be 'BM,BN,BK,stages,warps', got {cfg}"
            )
            assert required_bk is not None, (
                f"bmm tile candidates given for shape {shape} but that shape has "
                f"no BLOCK_K pin in RLINF_SMALL_M_BMM_SHAPES, so bit-exactness "
                f"cannot be checked"
            )
            assert cfg[2] == required_bk, (
                f"bmm shape {shape} pins BLOCK_K={required_bk} (the value the "
                f"stock champion for that shape already uses, i.e. the fp32 "
                f"reduction order an unpatched build produces); candidate {cfg} "
                f"carries BLOCK_K={cfg[2]} and would silently break bit-exactness"
            )
    return cfgs


def install_small_m_bmm_configs() -> bool:
    """Widen inductor's bmm autotune space for the denoise attention BMMs.

    ``torch._inductor.kernel.bmm`` imported ``mm_configs`` by value, so
    :func:`install_small_m_mm_configs` never reached these two GEMMs; this
    patches ``bmm_configs`` itself. Idempotent; returns True when active, and
    fails soft (returns False) if this torch build moved the hook.
    """
    global _bmm_installed
    if _bmm_installed:
        return True
    if not small_m_bmm_enabled():
        return False

    try:
        # Lazy for the same circular-import reason as the mm patch.
        import torch._inductor.lowering  # noqa: F401
        from torch._inductor.kernel import bmm as bmm_kernel
        from torch._inductor.kernel import mm_common

        original = bmm_kernel.bmm_configs
        filtered_configs = mm_common.filtered_configs
    except Exception:  # pragma: no cover - torch internals moved
        return False

    shapes = _parse_bmm_shapes(os.environ.get("RLINF_SMALL_M_BMM_SHAPES"))
    cfgs = _parse_bmm_cfgs(os.environ.get("RLINF_SMALL_M_BMM_CFGS"), shapes)
    max_m = _env_int("RLINF_SMALL_M_BMM_MAX_M", 64)

    def _small_m_bmm_configs(m, n, k, *, device_type, **kwargs) -> Iterable:
        base = original(m, n, k, device_type=device_type, **kwargs)
        if device_type != "cuda":
            return base
        try:
            extra = cfgs.get((int(n), int(k))) if int(m) <= max_m else None
        except (TypeError, ValueError):  # symbolic shape -> leave it alone
            return base
        if not extra:
            return base
        # device_type is a bmm_configs-only argument; filtered_configs does not
        # take it.
        return itertools.chain(base, filtered_configs(m, n, k, configs=extra, **kwargs))

    bmm_kernel.bmm_configs = _small_m_bmm_configs
    _bump_cache_key_tag("small_m_bmm")

    _bmm_installed = True
    print(
        f"[pi05_infer] small-M bmm tiles installed: max_m={max_m} "
        + " ".join(f"{n}x{k}@BK{shapes[(n, k)]}={list(v)}" for (n, k), v in cfgs.items())
    )
    return True
