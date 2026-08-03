"""Extra small-M tile candidates for inductor's Triton ``mm``/``bmm`` templates.

The denoise loop runs every expert GEMM at ``M = action_chunk`` (50), so the two
weight-streaming projections of each layer -- ``down_proj`` (50x4096 @ 4096x1024)
and ``o_proj`` (50x2048 @ 2048x1024) -- are bandwidth problems whose only
parallelism is the tile grid.  Inductor's stock list has no ``BLOCK_M < 32``
entry and clamps ``BLOCK_M`` up to 64, which makes M a single tile: the grid is
``N / BLOCK_N = 32`` CTAs on a 110-SM card.  This module appends deeper-pipelined
small-``BLOCK_M`` candidates **for those two shapes only** and lets inductor's
autotuner pick.  The two attention BMMs are patched separately
(``install_small_m_bmm_configs``) because ``torch._inductor.kernel.bmm`` binds
``mm_configs`` by value at import time::

    P.V    bmm(8x50x1018, 8x1018x256)   -- extra candidates
    Q.K^T  bmm(8x50x256 , 8x256x1018)   -- pinned, see below

**Bit-exactness: measured, not derived.**  Safety is a property of the
``(shape, BLOCK_K, num_stages)`` combination and does not follow from any single
parameter.  Both plausible rules are false here: at K=4096 ``BLOCK_K`` is inert
while ``num_stages`` flips ``down_proj``'s output bits; at K=256 both are inert.
Every shipped entry is digest-verified equal to the unpatched build's output.
**Do not add a candidate on a parameter argument** -- re-run
``tools/bitexact_denoise_gemms.py`` / ``tools/bitexact_denoise_bmms.py``.

**Pinning ``Q.K^T``.**  Inductor cannot separate the top of this shape's field:
its nine best choices land within 0.0056-0.0062 ms and carry ``BLOCK_K`` 32, 64
and 128, so the winner -- and with it the fp32 reduction split -- moves from
process to process on a cold cache.  ``_DEFAULT_BMM_PINS`` replaces the candidate
list with the measured winner rather than appending to it.  The point is the
variance, not the mean: it takes the draw-to-draw spread to zero.

⚠️  **Verified on sm_120 only, and "bit-identical" is card-relative** -- it means
identical to what an unpatched build produces *on this card*, and an unpatched
build picks a different kernel elsewhere, so the reference itself moves.  The pin
declines to install off sm_120; the digest set only warns, because the env hooks
exist to explore.

**Scope.**  These patches replace names in ``torch._inductor.kernel.{mm,bmm}``,
so they are process-global: another model compiled in the same process whose GEMM
hits ``m <= 64`` with one of the listed ``(N, K)`` pairs gets the wider search
too, and its output bits relative to an unpatched run can move.  Narrow the
``*_SHAPES`` env var or turn the patch off if that matters.

Kill switches: ``RLINF_SMALL_M_MM=0``, ``RLINF_SMALL_M_BMM=0``,
``RLINF_SMALL_M_BMM_PIN=0`` (keep the appended candidates, un-pin).
Tuning: ``RLINF_SMALL_M_{MM,BMM}_{CFGS,SHAPES,MAX_M}``,
``RLINF_SMALL_M_BMM_PINS``; formats are in the parser docstrings below.

Per-tile measurements, the sweeps behind each choice and the digest runs are in
the (unpublished) internal record; the numbers they support are in README.md.
"""

import itertools
import os
import warnings
from typing import Iterable, Optional

__all__ = [
    "install_small_m_mm_configs",
    "small_m_mm_enabled",
    "install_small_m_bmm_configs",
    "small_m_bmm_enabled",
    "small_m_bmm_pin_enabled",
]

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps), all digest-verified.
# num_stages < 5 at BLOCK_K=128 flips down_proj's bits and must stay out.
# The BLOCK_N=32 / num_warps=4 corner is deliberately absent: filtered_configs
# clamps num_warps to BLOCK_M*BLOCK_N//256, and that corner is slower anyway.
_DEFAULT_CFGS = (
    (32, 32, 256, 4, 4),  # champion, both shapes
    (32, 32, 256, 3, 4),
    (16, 64, 128, 5, 4),  # previous champion; kept as the reference tile
    (32, 32, 128, 5, 4),
)

# (N, K) to widen the search for: the action expert's o_proj and down_proj only.
# Everything else -- prefix, adaRMS, action in/out -- keeps the stock list, so its
# kernel selection (and output bits) cannot move.
_DEFAULT_SHAPES = ((1024, 2048), (1024, 4096))

# Digest-verified for both mm shapes. Anything else via RLINF_SMALL_M_MM_CFGS is
# allowed but warns -- the hook exists to explore, and safety here is measured.
_VERIFIED_CFGS = frozenset(
    {
        (32, 32, 256, 4, 4),
        (32, 32, 256, 3, 4),
        (16, 64, 128, 5, 4),
        (32, 32, 128, 5, 4),
        (64, 32, 128, 5, 4),  # stock champion
    }
)

# ---- attention bmm ----------------------------------------------------------
# Shape allowlist. The value is the BLOCK_K the stock champion uses, kept as
# documentation only -- it used to be asserted as a bit-exactness pin, on the
# rule the module docstring records as false.
_DEFAULT_BMM_SHAPES = {
    (256, 1018): 128,  # P.V    bmm(8x50x1018, 8x1018x256)
    (1018, 256): 32,  # Q.K^T  bmm(8x50x256 , 8x256x1018)
}

# (N, K) -> the single tile inductor may use for that shape. Unlike
# _DEFAULT_BMM_CFGS this *replaces* the candidate list instead of appending to
# it: on Q.K^T the choice itself is the problem. sm_120 only -- see the module
# docstring. Digest-verified against every tile in the sweep.
_PIN_DEVICE_CAPABILITY = (12, 0)
_DEFAULT_BMM_PINS = {
    (1018, 256): (64, 128, 32, 4, 4),  # Q.K^T
}

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps) per (N, K).
_DEFAULT_BMM_CFGS = {
    # P.V. One entry, not a family: inductor times the three num_stages
    # variants identically and would pick between them by benchmark noise.
    # Widen with RLINF_SMALL_M_BMM_CFGS if another card wants a different corner.
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
        if cfg not in _VERIFIED_CFGS:
            warnings.warn(
                f"RLINF_SMALL_M_MM_CFGS: {cfg} is not in the digest-verified set "
                f"{sorted(_VERIFIED_CFGS)}. Bit-exactness of this tile is UNKNOWN -- "
                f"num_stages is known to change down_proj's fp32 accumulation order "
                f"at BLOCK_K=128 (18/36 outputs, max|d| 6.25e-2). Run "
                f"tools/bitexact_denoise_gemms.py with and without this config and "
                f"compare the digests before trusting any numerical-parity claim.",
                RuntimeWarning,
                stacklevel=2,
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
    _bump_cache_key_tag(_cfg_tag("small_m_mm", cfgs, shapes, max_m))

    _installed = True
    print(
        f"[pi05_infer] small-M mm tiles installed: shapes={list(shapes)} "
        f"max_m={max_m} extra_cfgs={list(cfgs)}"
    )
    return True


def _cfg_tag(name: str, *parts) -> str:
    """A cache-key tag that changes when the *candidate set* changes.

    A constant tag separates patch-on from patch-off but not one config list
    from another, so inductor replays the previously compiled winner and never
    benchmarks the new candidates -- which silently voids any A/B over tiles.
    """
    import hashlib

    digest = hashlib.sha256(repr(parts).encode()).hexdigest()[:12]
    return f"{name}:{digest}"


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
        for cfg in entries:
            assert len(cfg) == 5, (
                f"bmm tile entries must be 'BM,BN,BK,stages,warps', got {cfg}"
            )
        if shape not in shapes:
            warnings.warn(
                f"RLINF_SMALL_M_BMM_CFGS: shape {shape} is not in the swept set "
                f"{sorted(shapes)}. Nothing checks its bit-exactness -- run "
                f"tools/bitexact_denoise_bmms.py with and without these configs "
                f"and compare the digests.",
                RuntimeWarning,
                stacklevel=2,
            )
    return cfgs


def small_m_bmm_pin_enabled() -> bool:
    """True when the swept-shape tile pins should replace autotune's choice."""
    return _env_int("RLINF_SMALL_M_BMM_PIN", 1) != 0


def _parse_bmm_pins(raw: Optional[str]) -> dict:
    """``"NxK:BM,BN,BK,s,w;..."`` -> ``{(N, K): (BM, BN, BK, s, w)}``.

    One tile per shape, not a list: a pin whose list has two entries is not a
    pin. Passing an empty string disables pinning for that shape.
    """
    if raw is None:
        return dict(_DEFAULT_BMM_PINS)
    pins = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        nk, _, rest = entry.partition(":")
        n, _, k = nk.partition("x")
        assert rest and k, (
            f"RLINF_SMALL_M_BMM_PINS entries must be 'NxK:BM,BN,BK,stages,warps', "
            f"got {entry!r}"
        )
        cfg = tuple(int(v) for v in rest.split(","))
        assert len(cfg) == 5, (
            f"RLINF_SMALL_M_BMM_PINS: a pin is one tile "
            f"'BM,BN,BK,stages,warps', got {cfg}"
        )
        pins[(int(n), int(k))] = cfg
    return pins


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
    pins = _parse_bmm_pins(os.environ.get("RLINF_SMALL_M_BMM_PINS"))
    if not small_m_bmm_pin_enabled() or not _pin_device_matches():
        pins = {}

    def _small_m_bmm_configs(m, n, k, *, device_type, **kwargs) -> Iterable:
        base = original(m, n, k, device_type=device_type, **kwargs)
        if device_type != "cuda":
            return base
        try:
            shape = (int(n), int(k))
            small = int(m) <= max_m
        except (TypeError, ValueError):  # symbolic shape -> leave it alone
            return base
        if not small:
            return base
        # device_type is a bmm_configs-only argument; filtered_configs does not
        # take it.
        pin = pins.get(shape)
        if pin is not None:
            only = list(filtered_configs(m, n, k, configs=(pin,), **kwargs))
            # filtered_configs can rewrite or drop a tile (it clamps BLOCK_M to
            # next_power_of_2(m) and num_warps to BLOCK_M*BLOCK_N//256). If it
            # dropped this one there is nothing left to compile, so fall back
            # rather than hand inductor an empty candidate list.
            if only:
                return iter(only)
            return base
        extra = cfgs.get(shape)
        if not extra:
            return base
        return itertools.chain(base, filtered_configs(m, n, k, configs=extra, **kwargs))

    bmm_kernel.bmm_configs = _small_m_bmm_configs
    _bump_cache_key_tag(
        _cfg_tag(
            "small_m_bmm",
            sorted(cfgs.items()),
            sorted(shapes.items()),
            sorted(pins.items()),
            max_m,
        )
    )

    _bmm_installed = True
    appended = " ".join(f"{n}x{k}+={list(v)}" for (n, k), v in cfgs.items())
    pinned = " ".join(f"{n}x{k}:={list(v)}" for (n, k), v in pins.items()) or "none"
    print(
        f"[pi05_infer] small-M bmm tiles installed: max_m={max_m} "
        f"appended: {appended} pinned: {pinned}"
    )
    return True


def _pin_device_matches() -> bool:
    """True on the one card the tile sweep behind ``_DEFAULT_BMM_PINS`` was run on.

    A pin claims one tile beats all others *on this hardware*, which depends on
    SM count, L2 size and the roofline knee. Elsewhere, autotune is the better
    answer -- it is worse only where you have measured a replacement.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() == _PIN_DEVICE_CAPABILITY
    except Exception:  # pragma: no cover - no torch/CUDA is a decline, not a crash
        return False
