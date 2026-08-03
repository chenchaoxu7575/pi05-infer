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
r"""Extraction equivalence: does ``pi05_infer`` still compute what RLinf computes?

Loads both trees into ONE process, so there is one autotune state and one set of
cuBLAS handles and the cross-process drift that defeats ``--dump-actions`` cannot
enter::

    arm A = rlinf.models.embodiment.openpi.get_model   (site-packages transformers)
    arm B = pi05_infer.build_model                     (vendored pi05_infer.gemma)

Each arm runs twice as its own control: an arm that is not reproducible against
itself yields INCONCLUSIVE, never PASS.

Compared in order of directness: ``weights`` (sha256 per expert parameter, before
anything is computed), ``prefix/kv`` (same site-packages SigLIP on both arms, so
a difference means the extraction moved the prefix), ``step<i>`` (where the trees
actually differ), ``actions``.

Usage::

    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_extract python tools/bitexact_extraction.py \
      --out /tmp/bitexact_backfill/extraction.json
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
    p.add_argument(
        "--rlinf-root",
        default=os.environ.get("RLINF_ROOT"),
        required="RLINF_ROOT" not in os.environ,
        help="Checkout of RLinf to compare the extraction against.",
    )
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
    p.add_argument("--out", default=None)
    return p.parse_args()


def build_rlinf(args):
    from omegaconf import OmegaConf

    sys.path.insert(0, args.rlinf_root)
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create({
        "model_path": args.model_path,
        "precision": None,
        "openpi": {
            "config_name": args.config_name,
            "num_images_in_input": args.num_images,
            "noise_level": 0.5,
            "action_chunk": args.action_chunk,
            "num_steps": args.num_steps,
            "train_expert_only": True,
            "action_env_dim": args.action_dim,
            "noise_method": "flow_sde",
            "add_value_head": False,
            "value_after_vlm": False,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": True,
        },
        "openpi_data": None,
    })
    return get_model(cfg).to(args.device).eval()


def build_pi05(args):
    from pi05_infer import build_model

    return build_model(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=args.num_images,
        noise_level=0.5,
        action_chunk=args.action_chunk,
        num_steps=args.num_steps,
        train_expert_only=True,
        action_env_dim=args.action_dim,
        noise_method="flow_sde",
    ).to(args.device).eval()


def expert_weight_digest(model) -> str:
    exp = model.paligemma_with_expert.gemma_expert
    h = hashlib.sha256()
    for name, prm in sorted(exp.named_parameters()):
        h.update(name.encode())
        h.update(_sha(prm).encode())
    return h.hexdigest()[:16]


def instrument(model, rec: dict):
    """sha every stage of one seeded predict, without changing what is computed.

    Returns an ``uninstrument`` callable. Calling this twice without restoring in between
    would NEST the wrappers: the inner closure keeps counting, so the second run's steps
    land in the FIRST run's record as ``step10..step19`` and the same-arm control compares
    dicts with different key sets. That is a bug in the harness, not a real control
    failure -- but it looks exactly like one, so restore instead.
    """
    orig_prefix = model._build_prefix_cache  # noqa: SLF001
    orig_smvv = model.sample_mean_var_val
    orig_noise = model.sample_noise
    state = {"i": 0, "n": 0}

    def kv_pairs(pkv):
        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            return list(zip(pkv.key_cache, pkv.value_cache))
        return [(kv[0], kv[1]) for kv in pkv]

    def prefix(*a, **k):
        out = orig_prefix(*a, **k)
        _po, ppm, pkv = out
        pairs = kv_pairs(pkv)
        rec["prefix/pad"] = _sha(ppm)
        rec["prefix/kv"] = _sha_many([t for p in pairs for t in p])
        return out

    def smvv(x_t, idx, *a, **k):
        i = state["i"]
        state["i"] = i + 1
        rec[f"step{i}/x_in"] = _sha(x_t)
        out = orig_smvv(x_t, idx, *a, **k)
        rec[f"step{i}/mean"] = _sha(out[0])
        return out

    def noise(*a, **k):
        out = orig_noise(*a, **k)
        if state["n"] == 0:
            rec["noise0"] = _sha(out)
        state["n"] += 1
        return out

    model._build_prefix_cache = prefix  # noqa: SLF001
    model.sample_mean_var_val = smvv
    model.sample_noise = noise

    def uninstrument():
        model._build_prefix_cache = orig_prefix  # noqa: SLF001
        model.sample_mean_var_val = orig_smvv
        model.sample_noise = orig_noise

    return uninstrument


def main() -> int:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)
    print(f"gpu {torch.cuda.get_device_name(args.device)}  torch {torch.__version__}")

    from standalone_infer_bench import make_env_obs

    env_obs = make_env_obs(args)

    # --- arm A: RLinf reference -----------------------------------------------------
    print("\nbuilding arm A (RLinf reference)...")
    a = build_rlinf(args)
    if not args.no_compile:
        a.enable_torch_compile(mode=args.compile_mode)

    def call_a():
        out = a.predict_action_batch(env_obs, mode="eval")
        return out[0] if isinstance(out, tuple) else out

    # --- arm B: pi05_infer ----------------------------------------------------------
    print("building arm B (pi05_infer vendored)...")
    b = build_pi05(args)
    if not args.no_compile:
        b.enable_torch_compile(mode=args.compile_mode)

    def call_b():
        return b.predict_action_batch(env_obs)

    mods = {
        "A_expert_module": type(
            a.paligemma_with_expert.gemma_expert.model.layers[0]
        ).__module__,
        "B_expert_module": type(
            b.paligemma_with_expert.gemma_expert.model.layers[0]
        ).__module__,
        "A_prefix_module": type(
            a.paligemma_with_expert.paligemma.model.language_model.layers[0]
        ).__module__,
        "B_prefix_module": type(
            b.paligemma_with_expert.paligemma.model.language_model.layers[0]
        ).__module__,
    }
    print("\nisolation (the two arms MUST run different expert modules):")
    for k, v in mods.items():
        print(f"  {k:18s} {v}")
    arms_differ = mods["A_expert_module"] != mods["B_expert_module"]
    print(f"  -> expert modules differ: {arms_differ}")

    wA, wB = expert_weight_digest(a), expert_weight_digest(b)
    print(f"\nexpert weight digest  A {wA}   B {wB}   same={wA == wB}")

    for _ in range(args.warmup):
        with torch.no_grad():
            call_a()
            call_b()
    torch.cuda.synchronize()

    def seeded(model, call):
        rec: dict = {}
        undo = instrument(model, rec)
        try:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with torch.no_grad():
                actions = call()
            torch.cuda.synchronize()
        finally:
            undo()
        rec["actions"] = _sha(actions)
        return rec, actions.detach().clone()

    ra1, act_a1 = seeded(a, call_a)
    rb1, act_b1 = seeded(b, call_b)
    ra2, act_a2 = seeded(a, call_a)
    rb2, act_b2 = seeded(b, call_b)

    keys = [k for k in ra1 if k in rb1]
    assert set(ra1) == set(ra2) == set(rb1) == set(rb2), (
        f"the four records must have identical key sets; got "
        f"{sorted(set(ra1) ^ set(ra2))} / {sorted(set(rb1) ^ set(rb2))} -- "
        f"the instrumentation nested instead of being restored"
    )
    ctrl_a = all(ra1[k] == ra2.get(k) for k in ra1)
    ctrl_b = all(rb1[k] == rb2.get(k) for k in rb1)
    cross = all(ra1[k] == rb1[k] for k in keys)
    first_diff = next((k for k in keys if ra1[k] != rb1[k]), None)

    print("\nstage-by-stage (A = RLinf, B = pi05_infer):")
    for k in keys:
        mark = "OK " if ra1[k] == rb1[k] else "DIFF"
        print(f"  {mark} {k:14s} A {ra1[k]}  B {rb1[k]}")

    d = (act_a1.double() - act_b1.double()).abs()
    print(f"\nactions max|d| = {d.max().item():.2e}  "
          f"ndiff {(act_a1 != act_b1).sum().item()}/{act_a1.numel()}")
    print(f"control A vs A : {ctrl_a}")
    print(f"control B vs B : {ctrl_b}")

    if not arms_differ:
        v = "INCONCLUSIVE -- both arms resolved the same expert module, nothing was compared"
        rc = 2
    elif not (ctrl_a and ctrl_b):
        v = "INCONCLUSIVE -- an arm is not reproducible against itself in-process"
        rc = 2
    elif cross:
        v = "PASS (bit-exact end to end)"
        rc = 0
    else:
        v = f"FAIL -- first diverging stage: {first_diff}"
        rc = 1
    print(f"\nVERDICT: {v}")

    payload = {
        "verdict": v,
        "arms_differ": arms_differ,
        "modules": mods,
        "weights": {"A": wA, "B": wB, "same": wA == wB},
        "control_A": ctrl_a,
        "control_B": ctrl_b,
        "cross": cross,
        "first_diff": first_diff,
        "actions_max_abs": d.max().item(),
        "actions_ndiff": int((act_a1 != act_b1).sum().item()),
        "digests_A": ra1,
        "digests_B": rb1,
        "digests_A2": ra2,
        "digests_B2": rb2,
        "meta": {
            "compile_mode": "eager" if args.no_compile else args.compile_mode,
            "rlinf_root": args.rlinf_root,
            "model_path": args.model_path,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(args.device),
            "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
            "pid": os.getpid(),
        },
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
