"""Byte-level check that skipping the prefix LM's last layer leaves the KV cache alone.

The KV cache is the *only* thing ``sample_actions`` consumes from the prefix LM, so
"the cache is bit-identical" is the whole correctness argument for
``pi05_infer/prefix_last_layer.py`` -- stronger and far more direct than an
end-to-end action dump (``--dump-actions`` is not reproducible across processes in
max-autotune mode: two identical runs of the same arm already disagree by 5.4e-3,
see tools/bitexact_denoise_gemms.py).

This runs the real checkpoint through the real ``_build_prefix_cache`` and prints a
sha256 over every byte of all 18 layers' K and V.

Usage -- one process per arm, sharing one inductor cache dir so that the autotune
result cache pins every untouched decision::

    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_kv RLINF_SKIP_LAST_LM_LAYER=0 \
        python tools/bitexact_prefix_kv.py --out off.json
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_kv RLINF_SKIP_LAST_LM_LAYER=1 \
        python tools/bitexact_prefix_kv.py --out on.json

Always run an off-vs-off control first; only then compare off vs on. Compare with::

    python tools/bitexact_prefix_kv.py --compare a.json b.json
"""

import argparse
import hashlib
import json
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT",
        help="Checkpoint dir, same default as bench/standalone_infer_bench.py.",
    )
    parser.add_argument("--config-name", default="pi05_turtle")
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default="Press the button with the end-effector.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--no-compile", action="store_true", help="Skip torch.compile (eager)."
    )
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--out", default=None, help="Write the digests here as JSON.")
    parser.add_argument(
        "--compare",
        nargs=2,
        default=None,
        metavar=("A.json", "B.json"),
        help="Diff two digest files and exit.",
    )
    parser.add_argument(
        "--dump-kv",
        default=None,
        help="Also save the raw KV tensors here (.pt), for --delta.",
    )
    parser.add_argument(
        "--embs-file",
        default=None,
        help="Pin the LM input. Written on first use, replayed afterwards (embed_prefix "
        "is bypassed). Takes the SigLIP vision tower -- which is NOT reproducible across "
        "processes -- out of the comparison, so what is left is the LM alone.",
    )
    parser.add_argument(
        "--delta",
        nargs=2,
        default=None,
        metavar=("A.pt", "B.pt"),
        help="Report per-layer max|d| between two --dump-kv files and exit.",
    )
    return parser.parse_args()


def delta(path_a: str, path_b: str) -> int:
    """Size the disagreement, in absolute terms and in bfloat16 ULPs."""
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    print(f"A = {path_a}  skip_installed={a['skip_installed']}")
    print(f"B = {path_b}  skip_installed={b['skip_installed']}")
    ea, eb = a["prefix_embs"].float(), b["prefix_embs"].float()
    print(
        f"prefix_embs (LM input): max|d| {(ea - eb).abs().max().item():.3e}  "
        f"ndiff {(ea != eb).sum().item()}/{ea.numel()}"
    )
    print(
        f"{'layer':>6}{'max|dK|':>12}{'ulpK':>7}{'ndiffK%':>9}"
        f"{'max|dV|':>12}{'ulpV':>7}{'ndiffV%':>9}"
    )
    worst = 0.0
    for i, ((ka, va), (kb, vb)) in enumerate(zip(a["kv"], b["kv"])):
        row = [f"{i:6d}"]
        for x, y in ((ka, kb), (va, vb)):
            xf, yf = x.float(), y.float()
            d = (xf - yf).abs()
            mx = d.max().item()
            worst = max(worst, mx)
            # bfloat16 has 8 explicit mantissa bits: 1 ULP ~ 2^-8 of the magnitude.
            scale = torch.maximum(xf.abs(), yf.abs()).clamp_min(1e-30)
            ulp = (d / (scale * 2.0**-8)).max().item()
            row.append(f"{mx:12.3e}{ulp:7.1f}{100.0 * (xf != yf).float().mean():8.2f}%")
        print("".join(row))
    print(f"worst max|d| over all 36 tensors: {worst:.3e}")
    return 0


def _digest(t: torch.Tensor) -> str:
    """sha256 over the raw bytes of a tensor (dtype-agnostic, incl. bfloat16)."""
    flat = t.detach().contiguous().view(-1)
    raw = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()


def _kv_pairs(past_key_values):
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(kv[0], kv[1]) for kv in past_key_values]


def compare(path_a: str, path_b: str) -> int:
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    print(f"A = {path_a}  skip_installed={a['skip_installed']}")
    print(f"B = {path_b}  skip_installed={b['skip_installed']}")
    embs_same = a.get("prefix_embs") == b.get("prefix_embs")
    print(
        f"  prefix_embs (LM input, upstream of the patch)  "
        f"{'SAME' if embs_same else 'DIFF'}"
    )
    bad = 0
    for i, (ka, kb) in enumerate(zip(a["layers"], b["layers"])):
        for which in ("k", "v"):
            same = ka[which] == kb[which]
            if not same:
                bad += 1
            print(
                f"  layer {i:2d} {which}  {'SAME' if same else 'DIFF'}  "
                f"{ka[which][:16]} vs {kb[which][:16]}"
            )
    print(f"\ncombined  A={a['combined']}\n          B={b['combined']}")
    verdict = (
        "BIT-IDENTICAL" if a["combined"] == b["combined"] else f"{bad} tensors DIFFER"
    )
    print(f"VERDICT: {verdict}")
    return 0 if bad == 0 else 1


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.delta:
        return delta(*args.delta)

    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
    from standalone_infer_bench import make_env_obs

    from pi05_infer import build_model
    from pi05_infer.prefix_last_layer import ENV_VAR, skip_enabled

    model = build_model(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=args.num_images,
        noise_level=0.5,
        action_chunk=args.action_chunk,
        num_steps=args.num_steps,
        train_expert_only=True,
        action_env_dim=args.action_dim,
        noise_method="flow_sde",
    )
    model = model.to(args.device).eval()
    installed = bool(model._prefix_last_layer_skipped)  # noqa: SLF001
    print(
        f"{ENV_VAR}={os.environ.get(ENV_VAR, '(default)')} "
        f"skip_enabled={skip_enabled()} installed={installed}"
    )
    if not args.no_compile:
        model.enable_torch_compile(mode=args.compile_mode)

    captured = {}
    orig = model._build_prefix_cache  # noqa: SLF001
    orig_embed = model.embed_prefix

    pinned = None
    if args.embs_file and os.path.exists(args.embs_file):
        pinned = torch.load(args.embs_file, map_location=args.device)
        print(f"pinned LM input replayed from {args.embs_file}")

    def _capture_embed(*a, **kw):
        # out[0] = prefix_embs, the LM's input. Hashing it separates "my change moved
        # the KV" from "the vision tower / embedding was already different".
        out = tuple(pinned["embed_prefix"]) if pinned else orig_embed(*a, **kw)
        captured["prefix_embs"] = out[0].clone()
        captured["embed_prefix"] = tuple(
            t.clone() if torch.is_tensor(t) else t for t in out
        )
        return out

    def _capture(*a, **kw):
        prefix_output, pad_masks, pkv = orig(*a, **kw)
        captured["kv"] = [(k.clone(), v.clone()) for k, v in _kv_pairs(pkv)]
        captured["prefix_output_is_none"] = prefix_output is None
        return prefix_output, pad_masks, pkv

    model.embed_prefix = _capture_embed
    model._build_prefix_cache = _capture  # noqa: SLF001

    env_obs = make_env_obs(args)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    with torch.no_grad():
        model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    embs_digest = _digest(captured["prefix_embs"])
    kv_first = [(k.clone(), v.clone()) for k, v in captured["kv"]]
    # Second call, same process, same inputs: separates cross-process nondeterminism
    # from anything this patch does.
    with torch.no_grad():
        model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    repeat_stable = _digest(captured["prefix_embs"]) == embs_digest and all(
        _digest(a) == _digest(b)
        for (ka, va), (kb, vb) in zip(kv_first, captured["kv"])
        for a, b in ((ka, kb), (va, vb))
    )
    captured["kv"] = kv_first

    kv = captured["kv"]
    layers = []
    combined = hashlib.sha256()
    for k, v in kv:
        dk, dv = _digest(k), _digest(v)
        layers.append({"k": dk, "v": dv, "shape": list(k.shape), "dtype": str(k.dtype)})
        combined.update(dk.encode())
        combined.update(dv.encode())

    result = {
        "skip_installed": installed,
        # How many prefix layers got the fused q/k/v GEMM (pi05_infer/prefix_qkv_fused.py).
        # Recorded so the digest file says which arm produced it.
        "prefix_qkv_fused_layers": getattr(model, "_prefix_qkv_fused_layers", 0),
        "prefix_embs": embs_digest,
        "repeat_stable_in_process": repeat_stable,
        "prefix_output_is_none": captured["prefix_output_is_none"],
        "compile_mode": None if args.no_compile else args.compile_mode,
        "num_layers": len(kv),
        "kv_shape": list(kv[0][0].shape),
        "layers": layers,
        "combined": combined.hexdigest(),
    }
    for i, layer in enumerate(layers):
        print(f"  layer {i:2d}  k {layer['k']}  v {layer['v']}")
    print(f"kv shape {result['kv_shape']} dtype {layers[0]['dtype']}")
    print(f"PREFIX_EMBS {embs_digest}")
    print(f"repeat_stable_in_process={repeat_stable}")
    print(f"prefix_output_is_none={result['prefix_output_is_none']}")
    print(f"COMBINED {result['combined']}")

    if args.embs_file and pinned is None:
        torch.save({"embed_prefix": captured["embed_prefix"]}, args.embs_file)
        print(f"wrote {args.embs_file}")
    if args.dump_kv:
        torch.save(
            {
                "skip_installed": installed,
                "prefix_embs": captured["prefix_embs"].cpu(),
                "kv": [(k.cpu(), v.cpu()) for k, v in kv],
            },
            args.dump_kv,
        )
        print(f"wrote {args.dump_kv}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
