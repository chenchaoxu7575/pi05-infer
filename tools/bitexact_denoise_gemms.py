"""Byte-level bit-exactness check for the two retiled denoise GEMMs.

``bench/standalone_infer_bench.py --dump-actions`` cannot certify this change:
MEASURED on RTX PRO 5000, two *identical* runs of the same arm (same code, same
inductor cache, same seed) already disagree in 300/300 action elements with
``max|d| = 5.4e-3`` -- larger than the off-vs-on difference. Something upstream
of the denoise loop (cuBLAS/cuDNN per-process algorithm selection in the SigLIP
vision tower is the prime suspect) is not reproducible across processes, so the
end-to-end dump has no resolving power at the bit level.

This tool tests what the change actually touches instead: it drives the real
checkpoint's ``o_proj`` and ``down_proj`` weights for all 18 expert layers
through ``torch.compile``, at the production shapes, dtypes and *strides* --
including the transposed-then-cloned attention output that inductor fuses into
``triton_tem_fused_clone_mm_8`` -- and prints a sha256 over every output byte.

Usage (one process per arm, sharing one inductor cache dir so that the autotune
result cache pins every untouched decision -- see tools/ab_small_m_mm.sh)::

    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_MM=0 python tools/bitexact_denoise_gemms.py
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_be RLINF_SMALL_M_MM=1 python tools/bitexact_denoise_gemms.py

The two runs must print the same digest.
"""

import argparse
import hashlib
import os

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,
        help="Checkpoint dir, same default as bench/standalone_infer_bench.py.",
    )
    parser.add_argument("--config-name", default="pi05_turtle")
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _down_proj(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def _o_proj(attn: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # attn arrives as [B, Hq, M, D] from SDPA; the model transposes, reshapes and
    # clones before o_proj, and inductor folds that clone into the mm prologue.
    b, hq, m, d = attn.shape
    return torch.nn.functional.linear(
        attn.transpose(1, 2).reshape(b, m, hq * d).contiguous(), w
    )


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    from pi05_infer import build_model
    from pi05_infer.inductor_mm_tiles import (
        install_small_m_mm_configs,
        small_m_mm_enabled,
    )

    installed = install_small_m_mm_configs()
    print(
        f"RLINF_SMALL_M_MM={os.environ.get('RLINF_SMALL_M_MM', '(default)')} "
        f"enabled={small_m_mm_enabled()} installed={installed}"
    )

    model = build_model(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=3,
        noise_level=0.5,
        action_chunk=args.action_chunk,
        num_steps=10,
        train_expert_only=True,
        action_env_dim=6,
        noise_method="flow_sde",
    )
    model = model.to(args.device).eval()
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    assert len(layers) > 0, "action expert has no layers; model wiring changed"

    # max-autotune-no-cudagraphs: the tile selection is identical to production
    # (same autotune cache entries), but nothing is captured into a CUDA graph, so
    # this check cannot be perturbed by graph-pool memory reuse.
    down = torch.compile(_down_proj, mode="max-autotune-no-cudagraphs", fullgraph=True)
    oproj = torch.compile(_o_proj, mode="max-autotune-no-cudagraphs", fullgraph=True)

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    m = args.action_chunk
    digest = hashlib.sha256()
    with torch.no_grad():
        for i, layer in enumerate(layers):
            wd = layer.mlp.down_proj.weight
            wo = layer.self_attn.o_proj.weight
            hq = model.paligemma_with_expert.gemma_expert.config.num_attention_heads
            d = wo.shape[1] // hq
            act = torch.randn(
                1, m, wd.shape[1], generator=g, device=args.device, dtype=wd.dtype
            )
            attn = torch.randn(
                1, hq, m, d, generator=g, device=args.device, dtype=wo.dtype
            )
            for out in (down(act, wd), oproj(attn, wo)):
                digest.update(out.cpu().view(torch.uint8).numpy().tobytes())
            if i == 0:
                print(
                    f"layer0 shapes: down {tuple(act.shape)} x {tuple(wd.shape)}  "
                    f"o {tuple(attn.shape)} x {tuple(wo.shape)}"
                )
    print(f"DIGEST {digest.hexdigest()}")


if __name__ == "__main__":
    main()
