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
"""Bit-exactness of the four "verified in eager only" optimizations, ON THE COMPILED PATH.

Ledger rows 1 (precomputed adaRMS modulation), 3 (fused QKV), 4 (static prefix-KV buffer)
and 5 (device-side ``att_masks``) were each verified off-vs-on **in eager, in one process**
(``adarms_cache_impl/test_*.py``, ``attmask_fix/verify_attmask.py``). None was ever checked
on the path that actually ships -- ``torch.compile(mode="max-autotune")`` -- and for row 1
that gap is not cosmetic: the modulation table is built by *eager* ``dense(cond)`` calls
while the baseline it replaces is inductor's ``triton_per_fused_addmm_0``. Two different
kernels computing the same GEMM are not obliged to agree in the last bit.

None of the four has an env-var kill switch (they are structural), so this tool turns each
one off from the outside, by monkeypatching the seam, and never edits the package:

  ``adarms``   ``_get_adarms_table`` -> ``None`` AND ``build_adarms_stack`` -> no-op, so
               ``sample_mean_var_val`` passes ``adarms_mod=None`` and every norm falls all
               the way back to its own ``dense(cond)``. Killing only the first would land on
               the Stage-A stacked GEMM, itself a 2.71e-3 change -- not the baseline.
  ``qkv``      ``GemmaModel.build_qkv_fused`` -> no-op, so ``qkv_fused_weight`` stays None
               and ``forward`` takes the three-skinny-GEMM branch.
  ``kvstatic`` ``GemmaModel.prime_kv_static`` -> no-op, so ``kv_static_k`` stays None and
               attention re-``torch.cat``s the 968-token prefix on every step.
  ``attmask``  ``embed_prefix`` rebuilds ``att_masks`` the pre-optimization way,
               ``torch.tensor(<python list>)[None, :].expand(...)`` (a host->device copy
               producing a stride-0 view instead of a materialised tensor).

Each patch is applied BEFORE ``enable_torch_compile``, so inductor traces the arm it is
supposed to trace, and the tool prints the resulting structural signature (fused weight
present or not, static buffer present or not, table present or not) as the mandatory
"the arms really are different" evidence (RESULTS_dump_actions_determinism.md §5 step 0).

``--freeze-prefix FILE`` writes the VLM prefix (pad mask + 18-layer KV) on first use and
replays it afterwards. That removes the SigLIP vision tower -- the one stage that is *not*
reproducible across processes -- from the comparison, which is what makes the off-vs-off
control clean enough for the gate to have any resolving power. Use it for ``adarms`` /
``qkv`` / ``kvstatic``, which live entirely downstream of the prefix. ``attmask`` is IN the
prefix, so it must be run without freezing.

Digests are taken per denoise step (``step<i>/x_in``, ``step<i>/mean``) as well as on the
final actions, so a divergence can be located instead of guessed at.

Usage -- always the control first::

    D=/workspace/rlinf_pub/bitexact_backfill/out/qkv
    for arm in off_a off_b on_a on_b; do
      case $arm in off_*) DIS=qkv ;; on_*) DIS=none ;; esac
      CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=/tmp/ti_bf \\
        python tools/bitexact_compiled_toggles.py --disable $DIS \\
          --freeze-prefix /tmp/prefix.pt --out $D/$arm.json
    done
    python tools/bitexact_compiled_toggles.py --verdict $D/off_a.json $D/off_b.json \\
                                                        $D/on_a.json $D/on_b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bench"))

TOGGLES = ("none", "adarms", "qkv", "kvstatic", "attmask")


def _sha(t) -> str:
    if t is None:
        return "none"
    if not torch.is_tensor(t):
        return f"nontensor:{type(t).__name__}"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT")
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
    p.add_argument("--stage1", action="store_true",
                   help="max-autotune-no-cudagraphs + the hand-captured denoise graph.")
    p.add_argument("--disable", choices=TOGGLES, default="none",
                   help="Which optimization to switch OFF for this run.")
    p.add_argument("--freeze-prefix", default=None,
                   help="Pin the VLM prefix (written on first use, replayed after).")
    p.add_argument("--save-actions", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--attmask-tensor-check", action="store_true",
                   help="In-process tensor-level check of ledger row 5; skips the e2e run.")
    p.add_argument("--verdict", nargs="+", default=None,
                   metavar="off_a.json off_b.json on_a.json on_b.json")
    return p.parse_args()


# ---------------------------------------------------------------------------------
# the four kill switches
# ---------------------------------------------------------------------------------
def apply_toggle(model, which: str) -> dict:
    """Turn one optimization off. Returns the structural signature of the resulting arm."""
    expert = model.paligemma_with_expert.gemma_expert.model

    if which == "adarms":
        # Two things have to go, not one. `_get_adarms_table -> None` makes
        # sample_mean_var_val pass adarms_mod=None -- but GemmaModel.forward's *next*
        # fallback is not the original 37 per-norm dense(cond) projections, it is the
        # Stage-A stacked GEMM (`adarms_Wstacked`, built unconditionally by
        # enable_torch_compile). Stage A was itself measured at 2.71e-3 against per-dense
        # (RESULTS_adarms_cache.md), so leaving it in would compare the table against
        # another optimization instead of against the baseline the ledger claims.
        # Killing build_adarms_stack too drops all the way through to `self.dense(cond)`,
        # which IS the pre-optimization path and IS what the eager test compared against.
        model._get_adarms_table = lambda *a, **k: None  # noqa: SLF001
        expert.build_adarms_stack = lambda *a, **k: None
        expert.adarms_Wstacked = None
        expert.adarms_bias_stacked = None
    elif which == "qkv":
        expert.build_qkv_fused = lambda *a, **k: None
        for layer in expert.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.qkv_fused_weight = None
                attn.build_qkv_fused = lambda *a, **k: None
    elif which == "kvstatic":
        expert.prime_kv_static = lambda *a, **k: None
        for layer in expert.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.kv_static_k = None
                attn.kv_static_v = None
                attn.prime_kv_static = lambda *a, **k: None
    elif which == "attmask":
        model.embed_prefix = _host_attmask_embed_prefix(model)
    elif which != "none":
        raise AssertionError(f"unknown toggle {which!r}")
    return {}


def _host_attmask_embed_prefix(model):
    """``embed_prefix`` with the pre-optimization host-side ``att_masks`` construction.

    Everything else (including the batched SigLIP call) is left exactly as shipped, so
    this isolates ledger row 5 alone. The original built a 1-D bool tensor from a python
    list via a synchronous H2D copy and then ``expand``ed it to [B, L] -- a stride-0 view,
    where the optimized form materialises a contiguous ``torch.zeros``.
    """
    import math

    def _embed_prefix(images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks, att_masks = [], [], []
        bsize = images[0].shape[0]
        all_embs = model.paligemma_with_expert.embed_image(
            torch.cat(images, dim=0) if len(images) > 1 else images[0]
        )
        num_img_embs = all_embs.shape[1]
        for view_idx, img_mask in enumerate(img_masks):
            embs.append(all_embs[view_idx * bsize : (view_idx + 1) * bsize])
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
        lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        # The pre-optimization form, verbatim (attmask_fix/attmask.diff):
        att = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att = att[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att

    return _embed_prefix


def attmask_tensor_check(model, args) -> dict:
    """Settle ledger row 5 at the tensor, not end-to-end.

    ``att_masks`` is upstream of the prefix KV, so it cannot be gated behind a frozen
    prefix -- and an unfrozen end-to-end gate under ``max-autotune`` has a ~4e-3 noise
    floor, i.e. no resolving power. But row 5 does not need one: ``att_masks`` is
    consumed by exactly two ops (``make_att_2d_masks`` and, via ``pad_masks``, the
    position ids), and everything after that is the same code on the same tensors. So if
    the 2-D and 4-D masks come out bit-identical *and* with identical dtype/shape/stride,
    the two arms are the same call with the same arguments from there on, and no
    end-to-end measurement can add anything.
    """
    import openpi.models.model as _model
    from standalone_infer_bench import make_env_obs

    from pi05_infer.openpi_patched.pi0_pytorch import make_att_2d_masks

    env_obs = make_env_obs(args)
    obs = model.obs_processor(env_obs)
    obs = model.input_transform(obs, transpose=False)
    obs = model.precision_processor(obs)
    obs = _model.Observation.from_dict(obs)
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _state = (
            model._preprocess_observation(obs, train=False)  # noqa: SLF001
        )
        shipped = model.embed_prefix
        _e1, pad1, att1 = shipped(images, img_masks, lang_tokens, lang_masks)
        _e2, pad2, att2 = _host_attmask_embed_prefix(model)(
            images, img_masks, lang_tokens, lang_masks
        )
        m2d_1 = make_att_2d_masks(pad1, att1)
        m2d_2 = make_att_2d_masks(pad2, att2)
        m4d_1 = model._prepare_attention_masks_4d(m2d_1)  # noqa: SLF001
        m4d_2 = model._prepare_attention_masks_4d(m2d_2)  # noqa: SLF001
        pos1 = torch.cumsum(pad1, dim=1) - 1
        pos2 = torch.cumsum(pad2, dim=1) - 1
    torch.cuda.synchronize()

    def cmp(name, a, b):
        r = {
            "equal": bool(torch.equal(a, b)),
            "shape_a": list(a.shape), "shape_b": list(b.shape),
            "stride_a": list(a.stride()), "stride_b": list(b.stride()),
            "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
            "sha_a": _sha(a), "sha_b": _sha(b),
        }
        print(f"  {name:12s} equal={r['equal']}  shape {r['shape_a']}  "
              f"stride device={r['stride_a']} host={r['stride_b']}  dtype {r['dtype_a']}")
        return r

    print("== att_masks (ledger row 5): device-built vs host-built, in one process ==")
    out = {
        "att_masks": cmp("att_masks", att1, att2),
        "att_2d": cmp("att_2d", m2d_1, m2d_2),
        "att_4d": cmp("att_4d", m4d_1, m4d_2),
        "position_ids": cmp("position_ids", pos1, pos2),
        "pad_masks": cmp("pad_masks", pad1, pad2),
    }
    # The raw att_masks may legitimately differ in STRIDE (expand vs materialise) while
    # being value-identical; what has to match bit-for-bit is what the model consumes.
    consumed_ok = all(out[k]["equal"] for k in ("att_2d", "att_4d", "position_ids"))
    same_layout = (
        out["att_2d"]["stride_a"] == out["att_2d"]["stride_b"]
        and out["att_4d"]["stride_a"] == out["att_4d"]["stride_b"]
    )
    print(f"  consumed tensors bit-identical: {consumed_ok}   identical layout: {same_layout}")
    out["consumed_bit_identical"] = consumed_ok
    out["consumed_same_layout"] = same_layout
    return out


def arm_signature(model) -> dict:
    """Structural proof that the arm is the arm it claims to be (step 0 of the protocol)."""
    expert = model.paligemma_with_expert.gemma_expert.model
    attn0 = expert.layers[0].self_attn
    return {
        "qkv_fused_layers": sum(
            1 for lyr in expert.layers
            if getattr(getattr(lyr, "self_attn", None), "qkv_fused_weight", None) is not None
        ),
        "qkv_fused_shape": (
            list(attn0.qkv_fused_weight.shape)
            if getattr(attn0, "qkv_fused_weight", None) is not None else None
        ),
        "kv_static_layers": sum(
            1 for lyr in expert.layers
            if getattr(getattr(lyr, "self_attn", None), "kv_static_k", None) is not None
        ),
        "kv_static_shape": (
            list(attn0.kv_static_k.shape)
            if getattr(attn0, "kv_static_k", None) is not None else None
        ),
        "adarms_stack_shape": (
            list(expert.adarms_Wstacked.shape)
            if getattr(expert, "adarms_Wstacked", None) is not None else None
        ),
        "adarms_table_shape": (
            list(model._adarms_table.shape)  # noqa: SLF001
            if getattr(model, "_adarms_table", None) is not None else None
        ),
        "embed_prefix_fn": getattr(model.embed_prefix, "__qualname__", "?"),
    }


# ---------------------------------------------------------------------------------
def run(args) -> dict:
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    import standalone_infer_bench as B
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

    if args.attmask_tensor_check:
        if not args.no_compile:
            model.enable_torch_compile(mode=args.compile_mode)
        res = attmask_tensor_check(model, args)
        return {"attmask_tensor_check": res, "digests": {}, "signature": {},
                "meta": {"disable": "attmask", "compile_mode":
                         "eager" if args.no_compile else args.compile_mode,
                         "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(args.device)}}

    # Turn the optimization off BEFORE compile, so inductor traces this arm.
    apply_toggle(model, args.disable)

    mode = args.compile_mode
    if args.stage1:
        mode = B.resolve_compile_mode(args)
    if not args.no_compile:
        model.enable_torch_compile(mode=mode)
    if args.stage1:
        model.capture_cuda_graph(args.batch_size, args.batch_size)
        assert model.is_cuda_graph_enabled(), (
            "--stage1 requested but capture_cuda_graph() did not install a manager."
        )

    env_obs = make_env_obs(args)
    rec: dict = {}
    step = {"i": 0}

    orig_prefix = model._build_prefix_cache  # noqa: SLF001
    orig_smvv = model.sample_mean_var_val
    orig_noise = model.sample_noise

    blob = None
    if args.freeze_prefix and os.path.exists(args.freeze_prefix):
        blob = torch.load(args.freeze_prefix, map_location=args.device)

    def prefix(*a, **k):
        out = orig_prefix(*a, **k)
        po, ppm, pkv = out
        if blob is not None:
            ppm.copy_(blob["pad"])
            for (kk, vv), (sk, sv) in zip(model._denoise_kv_pairs(pkv), blob["kv"]):  # noqa: SLF001
                kk.copy_(sk)
                vv.copy_(sv)
        elif args.freeze_prefix:
            torch.save(
                {"pad": ppm.detach().clone().cpu(),
                 "kv": [(k_.detach().clone().cpu(), v_.detach().clone().cpu())
                        for k_, v_ in model._denoise_kv_pairs(pkv)]},  # noqa: SLF001
                args.freeze_prefix,
            )
        pairs = model._denoise_kv_pairs(pkv)  # noqa: SLF001
        rec.setdefault("prefix/pad", _sha(ppm))
        rec.setdefault("prefix/kv", _sha_many([t for p in pairs for t in p]))
        return po, ppm, pkv

    def smvv(x_t, idx, *a, **k):
        i = step["i"]
        step["i"] = i + 1
        rec[f"step{i}/x_in"] = _sha(x_t)
        out = orig_smvv(x_t, idx, *a, **k)
        rec[f"step{i}/mean"] = _sha(out[0])
        return out

    noise_n = {"n": 0}

    def noise(*a, **k):
        out = orig_noise(*a, **k)
        if noise_n["n"] == 0:
            rec["noise0"] = _sha(out)
        noise_n["n"] += 1
        return out

    model._build_prefix_cache = prefix  # noqa: SLF001
    model.sample_mean_var_val = smvv
    model.sample_noise = noise

    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    if args.stage1:
        B.verify_stage1(model)
    rec.clear()
    step["i"] = 0
    noise_n["n"] = 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        actions = model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    rec["actions"] = _sha(actions)

    sig = arm_signature(model)
    print(f"disable={args.disable}  stage1={args.stage1}  mode={'eager' if args.no_compile else mode}")
    print(f"arm signature: {json.dumps(sig)}")
    print(f"frozen prefix: {'replayed' if blob is not None else ('written' if args.freeze_prefix else 'no')}")
    for k in ("noise0", "prefix/pad", "prefix/kv", "step0/mean", "step9/mean", "actions"):
        if k in rec:
            print(f"  {k:14s} {rec[k]}")

    if args.save_actions:
        torch.save(actions.detach().cpu(), args.save_actions)
    return {
        "digests": rec,
        "signature": sig,
        "meta": {
            "disable": args.disable,
            "stage1": bool(args.stage1),
            "compile_mode": "eager" if args.no_compile else mode,
            "frozen_prefix": bool(blob is not None),
            "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(args.device),
            "pid": os.getpid(),
        },
    }


def verdict(paths) -> int:
    """off_a off_b on_a on_b -> control first, cross-arm only if the control is clean."""
    assert len(paths) == 4, "need exactly four runs: off_a off_b on_a on_b"
    runs = []
    for p in paths:
        with open(p) as fh:
            runs.append(json.load(fh))
    names = ["off_a", "off_b", "on_a", "on_b"]
    keys = list(runs[0]["digests"])

    print("arm signatures (the arms MUST differ here, or the gate is testing nothing):")
    for n, r in zip(names, runs):
        print(f"  {n:6s} disable={r['meta']['disable']:9s} {json.dumps(r['signature'])}")
    sig_off = json.dumps(runs[0]["signature"], sort_keys=True)
    sig_on = json.dumps(runs[2]["signature"], sort_keys=True)
    arms_differ = sig_off != sig_on
    print(f"  -> off and on signatures differ: {arms_differ}")

    def eq(a, b):
        return all(runs[a]["digests"].get(k) == runs[b]["digests"].get(k) for k in keys)

    def first_diff(a, b):
        for k in keys:
            if runs[a]["digests"].get(k) != runs[b]["digests"].get(k):
                return k
        return None

    ctrl_off, ctrl_on = eq(0, 1), eq(2, 3)
    cross = eq(0, 2) and eq(1, 3)
    print(f"\ncontrol off_a vs off_b : {ctrl_off}" + ("" if ctrl_off else f"  (first diff {first_diff(0,1)})"))
    print(f"control on_a  vs on_b  : {ctrl_on}" + ("" if ctrl_on else f"  (first diff {first_diff(2,3)})"))
    print(f"cross   off   vs on    : {cross}" + ("" if cross else f"  (first diff {first_diff(0,2)})"))

    if not arms_differ:
        print("\nVERDICT: INCONCLUSIVE -- the two arms have identical structural signatures, "
              "so the toggle did not take effect and the comparison tests nothing.")
        return 2
    if not (ctrl_off and ctrl_on):
        print("\nVERDICT: INCONCLUSIVE -- the same-arm control is not bit-exact, so the "
              "cross-arm comparison carries no information.")
        return 2
    print(f"\nVERDICT: {'PASS (bit-exact)' if cross else 'FAIL -- the arms really do differ'}")
    return 0 if cross else 1


def main() -> int:
    args = parse_args()
    if args.verdict:
        return verdict(args.verdict)
    payload = run(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
