# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Does batching the three camera views into ONE SigLIP call change the numbers?

``embed_prefix`` (``pi05_infer/openpi_patched/pi0_pytorch.py``) runs the vision tower
once over ``torch.cat(images, dim=0)`` instead of once per view. The historical record
(``HANDOFF_min_infer_repro.md:11``) reports "action max diff 4.9e-3" for this change and
attributes it to "cuBLAS kernel-selection noise" -- but 4.9e-3 is the same order as the
*end-to-end gate's own* cross-process noise floor (2.5-5.4e-3), and that noise floor also
lives in the SigLIP tower, so the end-to-end number cannot separate the two.

This tool settles it **inside one process**, at the tensor the change actually produces:

* ``VIEW`` -- ``embed_image(cat(v0,v1,v2))`` sliced back per view, vs
  ``[embed_image(v) for v in views]``. This is the whole change, one call apart.
* ``KV``   -- the same question propagated through the full prefix: the 18-layer
  PaliGemma KV cache is the ONLY thing ``sample_actions`` consumes from the prefix.
* ``ACT``  -- the [1, 50, 6] actions, as a cross-check only.

Both arms run in the same process against the same weights, so cross-process autotune
drift (RESULTS_dump_actions_determinism.md) cannot reach the comparison. Each arm is
additionally run twice, giving an **in-process control**: if arm-vs-itself is not
bit-identical the tool prints INCONCLUSIVE and never PASS.

Prior expectation, before measuring: LayerNorm, GELU and the residual adds are per-sample
ops and SigLIP attention never crosses the batch dimension, so batching is *mathematically*
an identity. It is not automatically a *bit-level* identity: the GEMMs go from M=256 to
M=768 and the LayerNorm reductions from xnumel=256 to xnumel=768, so cuBLAS/inductor are
free to pick different tile shapes, split-k and reduction splits, each of which reorders
fp32 accumulation. So a nonzero result here means "the batched path is a different but
equally valid summation order", not "the batched path is wrong" -- and a zero result
means the two happened to land on the same order.

Usage (one process, both arms)::

    CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=/tmp/ti_siglip \\
      /opt/venv/openpi/bin/python tools/bitexact_siglip_batch.py --out /tmp/siglip.json
    ... --no-compile     # eager, for the mathematical-identity question alone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"),
)


def _sha(t) -> str:
    if t is None:
        return "none"
    a = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(a.shape)).encode())
    h.update(str(a.dtype).encode())
    h.update(a.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def _sha_many(ts) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(_sha(t).encode())
    return h.hexdigest()[:16]


def _delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Absolute delta plus the same delta measured in ULPs of the tensor's own dtype.

    Absolute numbers are meaningless without the scale: a bf16 value of ~600 has a
    1-ULP spacing of 2.3, so "max|d| = 2.5" can be either a catastrophic error or a
    single rounding step depending on where it lands. bf16 keeps 8 total mantissa
    bits (7 explicit), fp32 keeps 24 -- so 1 ULP is 2^-8 / 2^-24 of the magnitude.
    """
    x, y = a.detach().float(), b.detach().float()
    d = (x - y).abs()
    bits = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}.get(
        a.dtype, 24
    )
    scale = torch.maximum(x.abs(), y.abs()).clamp_min(1e-30)
    ulp = d / (scale * 2.0 ** -bits)
    return {
        "max_abs": d.max().item(),
        "max_ulp": ulp.max().item(),
        "mean_ulp": ulp.mean().item(),
        "p9999_ulp": torch.quantile(ulp.flatten().float(), 0.9999).item(),
        "max_rel": (d / scale).max().item(),
        "absmax": x.abs().max().item(),
        "dtype": str(a.dtype),
        "ndiff": int((x != y).sum().item()),
        "numel": int(x.numel()),
        "equal": bool(torch.equal(a.detach(), b.detach())),
    }


def _fmt(d: dict) -> str:
    return (
        f"equal={d['equal']}  max|d|={d['max_abs']:.3e}  "
        f"max={d['max_ulp']:.2f} ULP  mean={d['mean_ulp']:.3f} ULP  "
        f"|x|max={d['absmax']:.3e}  ndiff={d['ndiff']}/{d['numel']}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,)
    p.add_argument("--config-name", default="pi05_turtle")
    p.add_argument("--action-chunk", type=int, default=50)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt", default="Press the button with the end-effector.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="max-autotune")
    p.add_argument("--skip-e2e", action="store_true", help="Skip the ACT stage.")
    p.add_argument("--no-depth-profile", action="store_true",
                   help="Skip the per-encoder-layer ULP profile (forward hooks).")
    p.add_argument("--out", default=None)
    return p.parse_args()


def looped_embed_prefix(model):
    """``embed_prefix`` with the pre-optimization per-view ViT loop.

    Byte-for-byte the upstream openpi body except that the batched
    ``embed_image(torch.cat(images, 0))`` is replaced by one call per view -- and the
    device-side ``att_masks`` is kept, so this isolates SigLIP batching alone.
    """
    import math

    def _embed_prefix(images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks = [], []
        bsize = images[0].shape[0]
        num_img_embs = None
        for img, img_mask in zip(images, img_masks):
            img_emb = model.paligemma_with_expert.embed_image(img)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.zeros(
            pad_masks.shape[0], pad_masks.shape[1], dtype=torch.bool,
            device=pad_masks.device,
        )
        return embs, pad_masks, att_masks

    return _embed_prefix


def main() -> int:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    from standalone_infer_bench import make_env_obs

    from pi05_infer import build_model

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
    if not args.no_compile:
        model.enable_torch_compile(mode=args.compile_mode)

    env_obs = make_env_obs(args)
    batched_embed_prefix = model.embed_prefix
    looped = looped_embed_prefix(model)

    # ---- warm both paths so nothing below pays a compile ---------------------------
    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    model.embed_prefix = looped
    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    model.embed_prefix = batched_embed_prefix
    torch.cuda.synchronize()

    # ---- reach the real, preprocessed model inputs ----------------------------------
    import openpi.models.model as _model

    obs = model.obs_processor(env_obs)
    obs = model.input_transform(obs, transpose=False)
    obs = model.precision_processor(obs)
    obs = _model.Observation.from_dict(obs)
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
            obs, train=False
        )
    print(f"views={len(images)} shape={tuple(images[0].shape)} dtype={images[0].dtype}")

    emb = model.paligemma_with_expert.embed_image
    res = {"meta": {
        "compile_mode": None if args.no_compile else args.compile_mode,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(args.device),
        "views": len(images),
        "pid": os.getpid(),
    }}

    # ================================ VIEW ==========================================
    def run_batched():
        with torch.no_grad():
            allv = emb(torch.cat(images, dim=0) if len(images) > 1 else images[0])
        torch.cuda.synchronize()
        b = images[0].shape[0]
        return [allv[i * b : (i + 1) * b].clone() for i in range(len(images))]

    def run_looped():
        with torch.no_grad():
            out = [emb(img).clone() for img in images]
        torch.cuda.synchronize()
        return out

    b1, b2 = run_batched(), run_batched()
    l1, l2 = run_looped(), run_looped()
    view = {
        "batched_a": [_sha(t) for t in b1],
        "batched_b": [_sha(t) for t in b2],
        "looped_a": [_sha(t) for t in l1],
        "looped_b": [_sha(t) for t in l2],
        "shape": list(b1[0].shape),
        "ctrl_batched": all(torch.equal(x, y) for x, y in zip(b1, b2)),
        "ctrl_looped": all(torch.equal(x, y) for x, y in zip(l1, l2)),
        "cross": [_delta(x, y) for x, y in zip(b1, l1)],
    }
    # Pairing control: a nonzero same-view delta only means "rounding" if the
    # WRONG-view delta is enormously larger. If they were the same size, the slicing
    # would be scrambling the views and the whole comparison would be meaningless.
    if len(images) > 1:
        view["mispair"] = _delta(b1[0], l1[1])
    res["VIEW"] = view
    print("\n== VIEW: SigLIP output, batched vs per-view loop ==")
    print(f"  in-process control  batched {view['ctrl_batched']}  looped {view['ctrl_looped']}")
    for i, d in enumerate(view["cross"]):
        print(f"  view{i}  {_fmt(d)}")
    if "mispair" in view:
        print("  pairing control (batched view0 vs looped view1 -- must be MUCH larger)")
        print(f"         {_fmt(view['mispair'])}")

    # ---- how the disagreement grows with SigLIP depth -------------------------------
    # Rounding-order divergence enters at the first reordered reduction and is then
    # amplified layer by layer. A wiring/aliasing bug would instead be full-scale at
    # the very first layer. The profile below tells the two apart.
    tower = None
    for path in ("paligemma.model.vision_tower", "paligemma.vision_tower"):
        obj = model.paligemma_with_expert
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            tower = obj
            break
        except AttributeError:
            continue
    if tower is not None and hasattr(tower, "vision_model") and not args.no_depth_profile:
      try:
        enc_layers = tower.vision_model.encoder.layers
        grabbed = {}

        def mk_hook(i):
            def _h(_m, _in, out):
                t = out[0] if isinstance(out, tuple) else out
                grabbed.setdefault(i, []).append(t.detach().clone())
            return _h

        handles = [layer.register_forward_hook(mk_hook(i)) for i, layer in enumerate(enc_layers)]
        with torch.no_grad():
            emb(torch.cat(images, dim=0) if len(images) > 1 else images[0])
        torch.cuda.synchronize()
        bat = {i: v[-1] for i, v in grabbed.items()}
        grabbed.clear()
        with torch.no_grad():
            for img in images:
                emb(img)
        torch.cuda.synchronize()
        loo = {i: torch.cat(v[-len(images):], dim=0) for i, v in grabbed.items()}
        for h in handles:
            h.remove()
        prof = []
        for i in sorted(bat):
            if bat[i].shape != loo[i].shape:
                continue
            prof.append({"layer": i, **_delta(bat[i], loo[i])})
        res["VIEW_depth_profile"] = prof
        print(f"  SigLIP encoder depth profile ({len(prof)} layers), max|d| in ULP:")
        for r in prof:
            if r["layer"] % 3 == 0 or r["layer"] == prof[-1]["layer"]:
                print(f"    layer {r['layer']:2d}  ndiff {r['ndiff']:>8d}/{r['numel']}  "
                      f"max {r['max_ulp']:8.2f} ULP  mean {r['mean_ulp']:7.3f} ULP")
      except Exception as exc:  # noqa: BLE001 - the profile is a diagnostic, not the gate
        print(f"  SigLIP depth profile unavailable: {type(exc).__name__}: {exc}")

    # ================================ KV ============================================
    cap = {}

    def digest_prefix(tag):
        with torch.no_grad():
            _po, pad, pkv = model._build_prefix_cache(  # noqa: SLF001
                images, img_masks, lang_tokens, lang_masks
            )
        torch.cuda.synchronize()
        pairs = model._denoise_kv_pairs(pkv)  # noqa: SLF001
        flat = [t.detach().clone() for p in pairs for t in p]
        cap[tag] = flat
        return {"pad": _sha(pad), "kv": _sha_many(flat), "nlayers": len(pairs),
                "kv_shape": list(pairs[0][0].shape)}

    model.embed_prefix = batched_embed_prefix
    kb1, kb2 = digest_prefix("b1"), digest_prefix("b2")
    model.embed_prefix = looped
    kl1, kl2 = digest_prefix("l1"), digest_prefix("l2")
    model.embed_prefix = batched_embed_prefix

    kv_ctrl_b = all(torch.equal(x, y) for x, y in zip(cap["b1"], cap["b2"]))
    kv_ctrl_l = all(torch.equal(x, y) for x, y in zip(cap["l1"], cap["l2"]))
    kv_cross = _delta(torch.cat([t.flatten() for t in cap["b1"]]),
                      torch.cat([t.flatten() for t in cap["l1"]]))
    res["KV"] = {"batched_a": kb1, "batched_b": kb2, "looped_a": kl1, "looped_b": kl2,
                 "ctrl_batched": kv_ctrl_b, "ctrl_looped": kv_ctrl_l, "cross": kv_cross}
    print("\n== KV: 18-layer prefix KV cache (the only thing denoise consumes) ==")
    print(f"  in-process control  batched {kv_ctrl_b}  looped {kv_ctrl_l}")
    print(f"  batched {kb1['kv']}   looped {kl1['kv']}")
    print(f"  cross  {_fmt(kv_cross)}")

    # ================================ ACT ===========================================
    if not args.skip_e2e:
        def seeded_actions():
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with torch.no_grad():
                a = model.predict_action_batch(env_obs)
            torch.cuda.synchronize()
            return a.detach().clone()

        model.embed_prefix = batched_embed_prefix
        ab1, ab2 = seeded_actions(), seeded_actions()
        model.embed_prefix = looped
        al1, al2 = seeded_actions(), seeded_actions()
        model.embed_prefix = batched_embed_prefix
        act = {
            "ctrl_batched": bool(torch.equal(ab1, ab2)),
            "ctrl_looped": bool(torch.equal(al1, al2)),
            "cross": _delta(ab1, al1),
            "batched_a": _sha(ab1), "looped_a": _sha(al1),
        }
        res["ACT"] = act
        print("\n== ACT: [1,50,6] actions (cross-check) ==")
        print(f"  in-process control  batched {act['ctrl_batched']}  looped {act['ctrl_looped']}")
        print(f"  cross  {_fmt(act['cross'])}")

    # ================================ verdict =======================================
    stages = [("VIEW", view["ctrl_batched"] and view["ctrl_looped"],
               all(d["equal"] for d in view["cross"])),
              ("KV", kv_ctrl_b and kv_ctrl_l, kv_cross["equal"])]
    if not args.skip_e2e:
        stages.append(("ACT", act["ctrl_batched"] and act["ctrl_looped"],
                       act["cross"]["equal"]))
    print("\n== VERDICT ==")
    rc = 0
    for name, ctrl, cross in stages:
        if not ctrl:
            v = "INCONCLUSIVE (in-process control failed)"
            rc = max(rc, 2)
        elif cross:
            v = "PASS (bit-identical)"
        else:
            v = "FAIL (the arms really do differ)"
            rc = max(rc, 1)
        print(f"  {name:5s} {v}")
        res.setdefault("verdict", {})[name] = v

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
