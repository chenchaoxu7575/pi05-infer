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
against the stock list.  The first round of this took ``BLOCK_M`` from 64 to 16
or 32 -- 64 CTAs instead of 32.  MEASURED in the model on RTX PRO 5000 Blackwell
(sm_120, 110 SM), nsys 2026.1.2, 12 predicts::

    down_proj  15.06 -> 11.71 us/call  (-22.2%, 591 -> 760 GB/s)
    o_proj      8.47 ->  6.94 us/call  (-18.1%, 530 -> 647 GB/s)
    denoise stream busy  11.48 -> 10.55 ms/predict; kernel counts unchanged
    e2e paired A/B, SM clock held equal: -0.88 ms/predict

A second round then widened ``BLOCK_K`` to 256, which the first round could not
reach because a module-level ``assert BLOCK_K == 128`` excluded the whole family
from the search -- see the bit-exactness note.  The champion today is
``(BLOCK_M, BLOCK_N, BLOCK_K, stages, warps) = (32, 32, 256, 4, 4)`` for both
shapes; per-call figures are with ``_DEFAULT_CFGS`` below.  **The two rounds have
separate baselines and do not chain.**

**Bit-exactness.**  An earlier version of this module claimed that ``BLOCK_K``
is the only tile parameter that changes the fp32 accumulation order and that
``BLOCK_M``/``BLOCK_N``/``num_warps``/``num_stages`` are numerically inert.
**Measurement refuted that, and in the dangerous direction** (2026-07-31, real
checkpoint weights, 18 layers x 2 ops, 3 seeds, 5.53 M bf16 elements)::

    down_proj (N=1024, K=4096), reference digest 2a25a0b5... :
        BLOCK_K 128 -> 256 (and -> 64, -> 512)   bit-identical
        num_stages 5 -> 4 at BLOCK_K=128         DIFFERS: 18/36 outputs,
                                                 359732/921600 elements,
                                                 max|d| 6.25e-2
        num_stages 5 -> 3 at BLOCK_K=128         same difference
        num_stages 4 or 3 at BLOCK_K=256         bit-identical
    (BLOCK_M/BLOCK_N are inert: (32,32,128,4,4) and (16,64,128,4,4) produce
     byte-identical output -- they differ only in stages from the reference.)

So the safe set is **not** derivable from any single parameter: it is a property
of the (shape, BLOCK_K, num_stages) combination and has to be measured.  The
list below therefore carries **only entries whose output digest was verified
equal to the shipping reference** for both shapes; the old
``assert BLOCK_K == 128`` gate was blocking the safe direction while waving the
unsafe one through.  ``o_proj`` (N=1024, K=2048) did not flip on any config
tested, so the constraint above is driven entirely by ``down_proj``.

⚠️  **"Bit-exact relative to what."**  Unpatched stock inductor is *itself* not
stable across fresh autotunes on ``mm(50x4096, 4096x1024)``: cuBLAS ``mm``
(0.0202 ms) and the Triton template (0.0204 ms) tie, and 1 of 4 cold caches
picked cuBLAS -- which has its own digest.  The reference this module preserves
is the Triton champion the shipped patch already selects.

**Scope.**  The patch replaces a name in ``torch._inductor.kernel.mm``, so it is
process-global: any *other* model compiled in the same process whose GEMM also
hits ``m <= 64`` with one of the listed ``(N, K)`` pairs gets the wider search
too.  That stays algebraically exact, but its kernel choice -- and therefore its
output bits relative to an unpatched run -- can move, and the digests below were
taken on *these* shapes with *these* weights, so they say nothing about another
model's.  There is no cheap version of this check: it is the whole reason the
list is short.  Narrow ``RLINF_SMALL_M_MM_SHAPES`` or set ``RLINF_SMALL_M_MM=0`` if
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
replays of an isolated probe on the two real bmm shapes::

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

**Pinning ``Q.K^T``.**  The paragraph above says not to *add* candidates to this
shape.  The reason it gives -- the winner moves from process to process on a
cold cache -- is also a reason to *remove* them.  MEASURED over 29 cold compiles
of unchanged source: autotune drew **5 different tiles** for this one shape, and
in-model the fastest and slowest of the five differ by **20.2%**.  Its own
selection margin has a **median of 0.00%** while its reproduction noise on the
same candidate is **6-24%** -- at that ratio it is not choosing, it is sampling.

``_DEFAULT_BMM_PINS`` therefore replaces the candidate list for this shape with
the single measured winner instead of appending to it.  Worth -0.106 ms/predict
in expectation, and it takes the draw-to-draw sd of 6.41 us/step to exactly
zero, which is what makes any later A/B on this shape readable at all.

**Bit-exactness for the BMMs.**  The old rule -- ``BLOCK_K`` decides the fp32
reduction order, everything else is inert -- was **refuted once on each of the
two shapes it was applied to, and in opposite directions**:

    down_proj, K=4096   BLOCK_K inert; num_stages FLIPS BITS at BLOCK_K=128
    Q.K^T,     K=256    BLOCK_K inert AND num_stages inert -- all 7 tiles swept
                        (18 layers x 7.33 M bf16 elements) are torch.equal

One shape says the rule is too weak, the other says it is too strong.  There is
no reformulation that covers both, because the answer depends on how the K loop
is actually split for that K -- **it can only be measured**.  So the per-shape
``BLOCK_K`` in ``_DEFAULT_BMM_SHAPES`` is now informational, the assert that
enforced it is gone, and both the appended candidates and the pin above carry
digests taken with ``tools/bitexact_denoise_bmms.py``.  A cuBLAS arm run as a
positive control *does* come out different, so that gate has resolving power.

⚠️  **``_VERIFIED_CFGS`` and the pin are verified per card, and "verified" stops
meaning anything on a different one.**  Both were measured on sm_120; the
reference they were compared against is that card's own stock champion, so on
another card the *reference itself* moves and the equality claim has no subject.
The pin declines to install off sm_120 for that reason (and because the tile
that wins there is a different question).  The digest set does not -- it only
warns -- because the env hook exists to explore.

Kill switch: ``RLINF_SMALL_M_BMM=0`` (whole patch),
``RLINF_SMALL_M_BMM_PIN=0`` (keep the appended candidates, restore autotune's
choice on the pinned shapes).
Tuning: ``RLINF_SMALL_M_BMM_SHAPES="NxK@BK;..."``,
``RLINF_SMALL_M_BMM_CFGS="NxK:BM,BN,BK,stages,warps|...;..."``,
``RLINF_SMALL_M_BMM_PINS="NxK:BM,BN,BK,stages,warps;..."``,
``RLINF_SMALL_M_BMM_MAX_M``.
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

# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).
#
# EVERY entry here is digest-verified equal to the shipping reference for BOTH
# shapes -- see the bit-exactness note above.  Do not add a candidate without
# re-running that check; "it only changes BLOCK_M/BLOCK_N/warps" is not a proof.
#
# (32,32,256,4,4) is the measured champion for both shapes (2026-07-31, in-model
# nsys at 1897 MHz): down_proj 14.794 -> 12.375 us/call (-16.4%), o_proj
# 9.414 -> 7.797 (-17.2%); denoise stream -0.654 ms/predict, e2e paired A/B
# -0.52 +/- 0.28 ms (4/4 same sign).
#
# Three entries were REMOVED on 2026-07-31 because they flip down_proj's output
# bits -- (16,64,128,4,4), (16,64,128,3,4), (16,32,128,4,2), all num_stages<5 at
# BLOCK_K=128.  They shipped for weeks and stayed bit-exact only because
# autotune happened to keep picking num_stages=5; (16,64,128,4,4) is ~1% faster
# than that champion, so a different autotune draw would have been both faster
# and silently non-reproducible.
#
# Deliberately NOT extended with the BLOCK_N=32 / num_warps=4 corner: filtered_
# configs clamps num_warps to BLOCK_M*BLOCK_N//256 (= 2 for a 16x32 tile), and
# forcing 4 warps past that clamp changes nothing (14.479 us either way).  The
# 128-CTA corner is reachable today and is simply slower.
_DEFAULT_CFGS = (
    (32, 32, 256, 4, 4),  # champion, both shapes
    (32, 32, 256, 3, 4),
    (16, 64, 128, 5, 4),  # previous champion; kept as the reference tile
    (32, 32, 128, 5, 4),
)

# (N, K) of the GEMMs to widen the search for: the action expert's o_proj and
# down_proj. Everything else -- the 968-token PaliGemma prefix, the adaRMS
# projections, the action in/out projections -- keeps inductor's stock list, so
# its kernel selection (and hence its output bits) cannot move.
_DEFAULT_SHAPES = ((1024, 2048), (1024, 4096))

# Digest-verified safe for both mm shapes (see the bit-exactness note). Anything
# outside this set supplied via RLINF_SMALL_M_MM_CFGS is allowed -- the env hook
# exists precisely to explore -- but it warns, because safety here is measured,
# not derivable from the tile parameters.
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
# (N, K) -> BLOCK_K the *stock* champion for that shape uses. This is the shape
# allowlist; the BLOCK_K is informational only. It used to be asserted as a
# bit-exactness pin, on the same rule the mm side refuted -- and on these two
# shapes the rule fails in *both* directions: at K=256, BLOCK_K and num_stages
# are alike inert (all 7 tiles swept produce byte-identical output). See the
# BMM bit-exactness note in the module docstring.
_DEFAULT_BMM_SHAPES = {
    (256, 1018): 128,  # P.V    bmm(8x50x1018, 8x1018x256)
    (1018, 256): 32,  # Q.K^T  bmm(8x50x256 , 8x256x1018)
}

# (N, K) -> the single tile inductor is allowed to use for that shape.
#
# Unlike _DEFAULT_BMM_CFGS, which *appends* candidates and leaves the choice to
# autotune, a pin *replaces* the candidate list: autotune has nothing left to
# choose. That is the point -- on Q.K^T the choice itself is the problem.
#
# MEASURED (2026-08-01/02, RTX PRO 5000, 29 cold compiles of the same source):
# autotune drew 5 different tiles for this one shape, and in-model the fastest
# and slowest of them differ by 20.2%. Inductor's own selection margin has a
# median of 0.00% while its reproduction noise on the same candidate is 6-24%,
# so it is not choosing -- it is sampling. Pinning is worth -0.106 ms/predict in
# expectation and, more usefully, drops the draw-to-draw sd of 6.41 us/step to
# exactly zero, which is what makes any later A/B on this shape readable.
#
# Numerically free: 7 tiles x 18 layers x 7.33 M bf16 elements, all torch.equal.
# The gate has resolving power -- a cuBLAS arm run as a positive control does
# come out different.
#
# ONLY APPLIED ON sm_120. The sweep was run on one card; on any other the tile
# that wins is a different question and autotune is the better answer, so the
# pin declines to install rather than pinning a tile nobody measured there.
_PIN_DEVICE_CAPABILITY = (12, 0)
_DEFAULT_BMM_PINS = {
    (1018, 256): (64, 128, 32, 4, 4),  # Q.K^T
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

    ``_bump_cache_key_tag("small_m_mm")`` used to pass a constant, which
    separates patch-on from patch-off but **not** one config list from another:
    editing ``_DEFAULT_CFGS`` or setting ``RLINF_SMALL_M_MM_CFGS`` left the key
    untouched, so inductor replayed the previously compiled winner and the new
    candidates were never benchmarked. Any A/B over tile candidates done that
    way is meaningless. Fold the actual inputs into the tag instead.
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

    A pin is a claim that one tile beats every other tile *on this hardware*.
    That claim does not travel: the winner depends on SM count, L2 size and the
    roofline knee, none of which are the same on another card. Everywhere else,
    decline to pin and let autotune do its job -- it is a worse answer only when
    you have measured a better one.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() == _PIN_DEVICE_CAPABILITY
    except Exception:  # pragma: no cover - no torch/CUDA is a decline, not a crash
        return False
