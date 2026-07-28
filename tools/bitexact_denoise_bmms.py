"""Byte-level bit-exactness check for the retiled denoise attention BMMs.

Companion to ``tools/bitexact_denoise_gemms.py``, which covers the two weight
projections; this one covers the two batched attention GEMMs that
``install_small_m_bmm_configs`` widens the autotune space for::

    Q.K^T   bmm(8x50x256 , 8x256x1018)   -> triton_tem_fused_bmm_5
    P.V     bmm(8x50x1018, 8x1018x256)   -> triton_tem_fused_bmm_7

Those exact shapes are what inductor reports for the production graph (see the
``AUTOTUNE bmm(...)`` lines in any ``max-autotune`` bench log): 8 query heads,
``action_chunk`` = 50 rows, head_dim 256, and 968 prefix + 50 suffix = 1018 KV
positions. Unlike the projections these GEMMs have no weights -- both operands
are activations -- so the inputs are generated, but they are generated in the
*shape the model produces*: one KV head broadcast to 8 query heads via
``repeat_kv`` (so the second operand keeps batch stride 0, as in production),
and the P matrix is a real softmax output rather than raw noise, because the
value distribution is what decides the rounding.

Why not ``--dump-actions``: two identical runs of the same arm already disagree
by ~5e-3 on the actions (per-process autotune of the SigLIP LayerNorm
reductions), so the end-to-end dump has no resolving power at the bit level.
See claude_mem/pi05_rollout_forward/RESULTS_dump_actions_determinism.md.

Usage (one process per arm, sharing one inductor cache dir so the autotune
result cache pins every untouched decision)::

    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_BMM=0 python tools/bitexact_denoise_bmms.py
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_BMM=1 python tools/bitexact_denoise_bmms.py

The two runs must print the same digest.
"""

import argparse
import hashlib
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=8, help="Query heads.")
    parser.add_argument("--kv-heads", type=int, default=1, help="KV heads (GQA).")
    parser.add_argument("--action-chunk", type=int, default=50, help="Rows (M).")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--kv-len", type=int, default=1018, help="968 prefix + 50 suffix tokens."
    )
    parser.add_argument(
        "--layers", type=int, default=18, help="Action-expert layers to sweep."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kvh, s, d = x.shape
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


def _qk(q: torch.Tensor, k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return torch.matmul(q, _repeat_kv(k, n_rep).transpose(2, 3))


def _pv(p: torch.Tensor, v: torch.Tensor, n_rep: int) -> torch.Tensor:
    return torch.matmul(p, _repeat_kv(v, n_rep))


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    from pi05_infer.inductor_mm_tiles import (
        install_small_m_bmm_configs,
        small_m_bmm_enabled,
    )

    installed = install_small_m_bmm_configs()
    print(
        f"RLINF_SMALL_M_BMM={os.environ.get('RLINF_SMALL_M_BMM', '(default)')} "
        f"enabled={small_m_bmm_enabled()} installed={installed}"
    )

    # The model sets this before compiling; it is part of the autotune choice key
    # (ALLOW_TF32), so leaving it at the default would benchmark a different
    # candidate set than production.
    torch.set_float32_matmul_precision("high")

    n_rep = args.heads // args.kv_heads
    assert n_rep * args.kv_heads == args.heads, (
        f"heads {args.heads} is not a multiple of kv_heads {args.kv_heads}"
    )

    qk = torch.compile(_qk, mode="max-autotune-no-cudagraphs", fullgraph=True)
    pv = torch.compile(_pv, mode="max-autotune-no-cudagraphs", fullgraph=True)

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    m, d, s = args.action_chunk, args.head_dim, args.kv_len
    dt = torch.bfloat16
    digest = hashlib.sha256()
    with torch.no_grad():
        for i in range(args.layers):
            q = torch.randn(
                1, args.heads, m, d, generator=g, device=args.device, dtype=dt
            )
            k = torch.randn(
                1, args.kv_heads, s, d, generator=g, device=args.device, dtype=dt
            )
            v = torch.randn(
                1, args.kv_heads, s, d, generator=g, device=args.device, dtype=dt
            )
            scores = qk(q, k, n_rep)
            # Production feeds P.V a softmax output, not raw scores.
            p = torch.softmax(scores.float() * (d**-0.5), dim=-1).to(dt)
            out = pv(p, v, n_rep)
            for t in (scores, out):
                digest.update(t.cpu().view(torch.uint8).numpy().tobytes())
            if i == 0:
                print(
                    f"layer0 shapes: qk {tuple(q.shape)} x {tuple(k.shape)} -> "
                    f"{tuple(scores.shape)}   pv {tuple(p.shape)} x {tuple(v.shape)} "
                    f"-> {tuple(out.shape)}"
                )
    print(f"DIGEST {digest.hexdigest()}")


if __name__ == "__main__":
    main()
