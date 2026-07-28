"""Fused Triton kernels for the pi0.5 action-expert denoise loop (RLinf).

Two matmul-**epilogue** fusions, each behind a ``torch.library`` custom op with a
plain-PyTorch fallback:

1. ``fused_gate_up_swiglu`` -- ``gelu_tanh(x @ Wg.T) * (x @ Wu.T)`` in one kernel
   (two accumulators over one shared A tile), replacing inductor's
   ``triton_tem_fused_mm_13`` + ``triton_tem_fused_gelu_mm_mul_14``.
2. ``fused_qkv_rope`` -- ``rope(x @ Wqkv.T)`` in one kernel, writing q into a
   ``[B, M, Hq, D]`` buffer and k/v straight into the static KV cache tails,
   replacing ``triton_tem_fused_mm_4`` + the three RoPE/store pointwise kernels.

**Bit-exactness.** Both kernels deliberately round each accumulator that the
unfused pipeline would have *stored as bf16* back through bf16 before the
elementwise epilogue runs (``acc.to(bf16).to(fp32)``).  That reproduces the
rounding of the unfused two-kernel path exactly, so the fusion is bit-exact
rather than merely algebraically equivalent -- provided the GEMM tiling matches
inductor's (``BLOCK_K`` fixed to the value inductor's template picked, since
``BLOCK_K`` is the only tile parameter that changes the fp32 accumulation order
over K).  ``BLOCK_M``/``BLOCK_N`` do not affect the result of any single output
element and are free tuning knobs.

Precision is unchanged everywhere: bf16 operands, fp32 accumulate, bf16 store.

Kill switches (both default to enabled; set to 0 to fall back to eager PyTorch):
``RLINF_FUSE_SWIGLU``, ``RLINF_FUSE_QKV_ROPE``.
Tuning: ``RLINF_FUSE_SWIGLU_CFG="BM,BN,BK,warps,stages"`` and
``RLINF_FUSE_QKV_CFG="BM,BN,BK,warps,stages"``.
"""

import os
from typing import Optional

import torch

__all__ = [
    "HAVE_FUSED_DENOISE_OPS",
    "fuse_swiglu_enabled",
    "fuse_qkv_rope_enabled",
    "fused_gate_up_swiglu",
    "fused_qkv_rope_kv",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_cfg(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name)
    if not raw:
        return default
    parts = tuple(int(p) for p in raw.split(","))
    assert len(parts) == len(default), (
        f"{name} must be {len(default)} comma-separated ints "
        f"(BM,BN,BK,warps,stages), got {raw!r}"
    )
    return parts


try:
    import triton
    import triton.language as tl

    # Same libdevice binding inductor's generated code uses, so `tanh` lowers to
    # the identical PTX and the gelu epilogue stays bit-exact.
    from torch._inductor.runtime.triton_helpers import libdevice

    HAVE_FUSED_DENOISE_OPS = True
except Exception:  # pragma: no cover - triton missing / too old
    HAVE_FUSED_DENOISE_OPS = False


# BLOCK_K values are pinned to what inductor's autotuner picked for these exact
# shapes on sm_120; changing them changes the fp32 reduction order over K and
# breaks bit-exactness against the unfused path (it stays algebraically exact).
_SWIGLU_CFG = _env_cfg("RLINF_FUSE_SWIGLU_CFG", (64, 32, 64, 4, 4))
_QKV_CFG = _env_cfg("RLINF_FUSE_QKV_CFG", (64, 16, 128, 4, 4))

# The SwiGLU fusion only pays off in the short-sequence (decode / denoise-suffix)
# regime it is tuned for: there the gate activation round-trip through HBM and the
# extra launch dominate. On a long prefix (M in the hundreds) that cost is
# amortised, the GEMM becomes compute-bound, and a single hand-fixed tile config
# loses to inductor's per-shape autotuned template. MEASURED: without this guard
# the fusion also fired on PaliGemma's 968-token prefix LM (which is a Gemma too)
# and cost +6.5 ms/predict there -- 5x more than the whole denoise-side win.
# Default = one M-tile, i.e. exactly the regime the kernel was written for.
_SWIGLU_MAX_M = _env_int("RLINF_FUSE_SWIGLU_MAX_M", _SWIGLU_CFG[0])


def fuse_swiglu_enabled() -> bool:
    """True when the fused gate/up+SwiGLU kernel should be used."""
    return HAVE_FUSED_DENOISE_OPS and _env_int("RLINF_FUSE_SWIGLU", 1) != 0


def fuse_qkv_rope_enabled() -> bool:
    """True when the fused QKV+RoPE kernel should be used."""
    return HAVE_FUSED_DENOISE_OPS and _env_int("RLINF_FUSE_QKV_ROPE", 1) != 0


if HAVE_FUSED_DENOISE_OPS:

    @triton.jit
    def _swiglu_mm_kernel(
        A,
        WG,
        WU,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """C = gelu_tanh(bf16(A @ WG.T)) * (A @ WU.T), one tile per program."""
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = rm % M
        rbn = rn % N
        rk = tl.arange(0, BLOCK_K)

        a_ptrs = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        g_ptrs = WG + (rbn[None, :] * stride_wn + rk[:, None] * stride_wk)
        u_ptrs = WU + (rbn[None, :] * stride_wn + rk[:, None] * stride_wk)

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            acc_g += tl.dot(a, tl.load(g_ptrs), allow_tf32=True)
            acc_u += tl.dot(a, tl.load(u_ptrs), allow_tf32=True)
            a_ptrs += BLOCK_K * stride_ak
            g_ptrs += BLOCK_K * stride_wk
            u_ptrs += BLOCK_K * stride_wk

        # Round the gate through bf16: the unfused path stores gate_proj's output
        # as bf16 and reads it back, so this keeps the result bit-identical.
        x = acc_g.to(tl.bfloat16).to(tl.float32)
        t3 = x * 0.5
        t5 = (x * x) * x
        t8 = x + t5 * 0.044715
        t11 = libdevice.tanh(t8 * 0.7978845608028654)
        out = (t3 * (t11 + 1.0)) * acc_u

        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(
            C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
            out.to(C.dtype.element_ty),
            mask,
        )

    @triton.jit
    def _qkv_rope_kernel(
        A,
        W,
        COS,
        SIN,
        QO,
        KO,
        VO,
        M,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_cb,
        stride_cm,
        stride_cd,
        stride_qb,
        stride_qm,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_km,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vm,
        stride_vd,
        krow0,
        HQ: tl.constexpr,
        HKV: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """QKV projection with RoPE applied in the epilogue.

        Programs on the rotated range own a *pair* of column tiles -- ``d0`` and
        ``d0 + D/2`` of the same head -- because RoPE couples exactly those two
        columns.  Both are accumulated from one shared A tile, so the pairing
        costs no extra activation traffic.
        """
        HALF: tl.constexpr = D // 2
        NT_ROT: tl.constexpr = HALF // BLOCK_N
        NT_V: tl.constexpr = D // BLOCK_N
        N_ROT: tl.constexpr = (HQ + HKV) * NT_ROT

        pid_n = tl.program_id(0)
        pid_bm = tl.program_id(1)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        b = pid_bm // grid_m
        pid_m = pid_bm % grid_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = rm % M
        rk = tl.arange(0, BLOCK_K)
        rn = tl.arange(0, BLOCK_N)
        mmask = (rm < M)[:, None]
        a_base = A + b * stride_ab + (ram[:, None] * stride_am + rk[None, :] * stride_ak)

        if pid_n < N_ROT:
            h = pid_n // NT_ROT
            t = pid_n % NT_ROT
            d0 = t * BLOCK_N
            nlo = h * D + d0
            lo_ptrs = W + ((nlo + rn)[None, :] * stride_wn + rk[:, None] * stride_wk)
            hi_ptrs = W + ((nlo + HALF + rn)[None, :] * stride_wn + rk[:, None] * stride_wk)
            a_ptrs = a_base
            acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                acc_lo += tl.dot(a, tl.load(lo_ptrs), allow_tf32=True)
                acc_hi += tl.dot(a, tl.load(hi_ptrs), allow_tf32=True)
                a_ptrs += BLOCK_K * stride_ak
                lo_ptrs += BLOCK_K * stride_wk
                hi_ptrs += BLOCK_K * stride_wk

            # bf16 round-trip == the value the unfused GEMM would have stored
            xlo = acc_lo.to(tl.bfloat16).to(tl.float32)
            xhi = acc_hi.to(tl.bfloat16).to(tl.float32)

            cs_base = b * stride_cb + ram[:, None] * stride_cm
            dl = (d0 + rn)[None, :] * stride_cd
            dh = (d0 + HALF + rn)[None, :] * stride_cd
            c_lo = tl.load(COS + cs_base + dl).to(tl.float32)
            s_lo = tl.load(SIN + cs_base + dl).to(tl.float32)
            c_hi = tl.load(COS + cs_base + dh).to(tl.float32)
            s_hi = tl.load(SIN + cs_base + dh).to(tl.float32)

            # rotate_half -> (-x_hi, x_lo); (q*cos) + (rotate_half(q)*sin)
            o_lo = xlo * c_lo - xhi * s_lo
            o_hi = xhi * c_hi + xlo * s_hi

            if h < HQ:
                qb = QO + b * stride_qb + h * stride_qh + rm[:, None] * stride_qm
                tl.store(
                    qb + (d0 + rn)[None, :] * stride_qd,
                    o_lo.to(QO.dtype.element_ty),
                    mmask,
                )
                tl.store(
                    qb + (d0 + HALF + rn)[None, :] * stride_qd,
                    o_hi.to(QO.dtype.element_ty),
                    mmask,
                )
            else:
                kb = (
                    KO
                    + b * stride_kb
                    + (h - HQ) * stride_kh
                    + (krow0 + rm[:, None]) * stride_km
                )
                tl.store(
                    kb + (d0 + rn)[None, :] * stride_kd,
                    o_lo.to(KO.dtype.element_ty),
                    mmask,
                )
                tl.store(
                    kb + (d0 + HALF + rn)[None, :] * stride_kd,
                    o_hi.to(KO.dtype.element_ty),
                    mmask,
                )
        else:
            j = pid_n - N_ROT
            hh = j // NT_V
            t = j % NT_V
            d0 = t * BLOCK_N
            n0 = (HQ + HKV) * D + hh * D + d0
            w_ptrs = W + ((n0 + rn)[None, :] * stride_wn + rk[:, None] * stride_wk)
            a_ptrs = a_base
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                acc += tl.dot(tl.load(a_ptrs), tl.load(w_ptrs), allow_tf32=True)
                a_ptrs += BLOCK_K * stride_ak
                w_ptrs += BLOCK_K * stride_wk
            vb = (
                VO
                + b * stride_vb
                + hh * stride_vh
                + (krow0 + rm[:, None]) * stride_vm
            )
            tl.store(
                vb + (d0 + rn)[None, :] * stride_vd,
                acc.to(VO.dtype.element_ty),
                mmask,
            )


# ---------------------------------------------------------------------------
# custom ops
# ---------------------------------------------------------------------------


def _swiglu_ref(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor) -> torch.Tensor:
    """Plain-PyTorch reference: the exact op sequence GemmaMLP.forward runs."""
    return (
        torch.nn.functional.gelu(
            torch.nn.functional.linear(x, wg), approximate="tanh"
        )
        * torch.nn.functional.linear(x, wu)
    )


@torch.library.custom_op("rlinf::gate_up_swiglu", mutates_args=())
def gate_up_swiglu(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor
) -> torch.Tensor:
    """gelu_tanh(x @ wg.T) * (x @ wu.T) in a single fused GEMM."""
    lead = x.shape[:-1]
    k = x.shape[-1]
    n = wg.shape[0]
    a = x.reshape(-1, k)
    m = a.shape[0]
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    bm, bn, bk, warps, stages = _SWIGLU_CFG
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _swiglu_mm_kernel[grid](
        a,
        wg,
        wu,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        wg.stride(0),
        wg.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )
    return out.view(*lead, n)


@gate_up_swiglu.register_fake
def _(x, wg, wu):
    return x.new_empty((*x.shape[:-1], wg.shape[0]))


@torch.library.custom_op("rlinf::qkv_rope_kv", mutates_args={"k_cache", "v_cache"})
def qkv_rope_kv(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    krow0: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused QKV projection + RoPE.

    Returns q laid out ``[B, M, Hq, D]`` (the caller transposes to
    ``[B, Hq, M, D]``, matching the unfused ``view().transpose()`` strides
    exactly so downstream kernels see identical layouts), and writes the rotated
    k / raw v straight into rows ``krow0 :`` of the static KV cache.
    """
    b, m, _ = x.shape
    d = head_dim
    q = torch.empty(b, m, n_q_heads, d, dtype=x.dtype, device=x.device)
    bm, bn, bk, warps, stages = _QKV_CFG
    half = d // 2
    n_rot = (n_q_heads + n_kv_heads) * (half // bn)
    n_v = n_kv_heads * (d // bn)
    grid = (n_rot + n_v, b * triton.cdiv(m, bm))
    cos3 = cos.reshape(b, m, d)
    sin3 = sin.reshape(b, m, d)
    _qkv_rope_kernel[grid](
        x,
        w,
        cos3,
        sin3,
        q,
        k_cache,
        v_cache,
        m,
        x.shape[-1],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        cos3.stride(0),
        cos3.stride(1),
        cos3.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        krow0,
        HQ=n_q_heads,
        HKV=n_kv_heads,
        D=d,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return q


@qkv_rope_kv.register_fake
def _(x, w, cos, sin, k_cache, v_cache, krow0, n_q_heads, n_kv_heads, head_dim):
    return x.new_empty((x.shape[0], x.shape[1], n_q_heads, head_dim))


# ---------------------------------------------------------------------------
# dispatchers (fast path + guards); these are what the model calls
# ---------------------------------------------------------------------------


def _rowmajor(w: torch.Tensor) -> bool:
    return w.dim() == 2 and w.stride(1) == 1 and w.stride(0) == w.shape[1]


def _needs_autograd(*tensors: torch.Tensor) -> bool:
    """True when this call would need a backward formula.

    Neither custom op registers one -- they exist for the inference/rollout path.
    Training (actor update, SFT) must keep the differentiable eager path, so the
    dispatchers bail out here rather than raising inside ``torch.library``.
    """
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def fused_gate_up_swiglu(
    x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor
) -> Optional[torch.Tensor]:
    """Fused SwiGLU, or ``None`` when the fast path does not apply."""
    if not fuse_swiglu_enabled() or _needs_autograd(x, wg, wu):
        return None
    bk = _SWIGLU_CFG[2]
    bn = _SWIGLU_CFG[1]
    m = 1
    for s in x.shape[:-1]:
        m *= s
    if (
        m > _SWIGLU_MAX_M
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or wg.dtype is not x.dtype
        or wu.dtype is not x.dtype
        or wg.shape != wu.shape
        or not _rowmajor(wg)
        or not _rowmajor(wu)
        or x.stride(-1) != 1
        or x.shape[-1] % bk
        or wg.shape[0] % bn
    ):
        return None
    return torch.ops.rlinf.gate_up_swiglu(x.contiguous(), wg, wu)


def fused_qkv_rope_kv(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    krow0: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Optional[torch.Tensor]:
    """Fused QKV+RoPE writing into the static KV cache, or ``None`` to fall back."""
    if not fuse_qkv_rope_enabled() or _needs_autograd(x, w):
        return None
    bm, bn, bk, _, _ = _QKV_CFG
    if (
        not x.is_cuda
        or x.dim() != 3
        or x.dtype not in (torch.bfloat16, torch.float16)
        or w.dtype is not x.dtype
        or not _rowmajor(w)
        or x.stride(-1) != 1
        or x.shape[-1] % bk
        or head_dim % (2 * bn)
        or w.shape[0] != (n_q_heads + n_kv_heads * 2) * head_dim
        or cos.shape[-1] != head_dim
        or cos.dtype is not x.dtype
        or k_cache is None
        or v_cache is None
        or k_cache.dtype is not x.dtype
        or krow0 + x.shape[1] != k_cache.shape[2]
    ):
        return None
    return torch.ops.rlinf.qkv_rope_kv(
        x.contiguous(),
        w,
        cos,
        sin,
        k_cache,
        v_cache,
        krow0,
        n_q_heads,
        n_kv_heads,
        head_dim,
    )
