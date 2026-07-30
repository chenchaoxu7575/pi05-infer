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
"""Same-process, kernel-level bit-exactness gate for the fused prefix QKV GEMM.

``tools/bitexact_gate.sh`` compares two *processes*, and under ``max-autotune`` two
processes running byte-identical code already disagree (coordinate-descent tuning
re-benchmarks per process). This gate has no such blind spot: it builds the model
**once**, runs the prefix unfused, installs
``pi05_infer/prefix_qkv_fused.install_fused_prefix_qkv`` on the live object, and
runs the identical input again. Nothing else in the process changed, so any
difference is the fusion and only the fusion.

What is compared
----------------
* Every layer's K and V, byte for byte. That is the entire prefix->denoise
  interface (``sample_actions`` consumes nothing else from the prefix LM).
* Because layer *i*'s K/V is a function of layer *i-1*'s full output, layers 1..17
  matching also proves ``q``, the attention, ``o_proj`` and the MLP are untouched.
* Layer 0's raw ``q``/``k``/``v`` are additionally hashed directly, via a forward
  hook on ``q_proj``/``k_proj``/``v_proj`` (unfused arm) versus a re-derivation
  from the fused weight (fused arm) -- the most direct possible statement of
  "concatenating along N changed nothing".

Runs eager by default so the comparison is not entangled with inductor autotuning.
``--compile-mode`` re-runs the same check on the compiled path; note that a
*recompile* happens between the arms there, so a difference in that mode is not
automatically attributable to the fusion.

Usage::

    python tools/bitexact_prefix_qkv.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT")
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
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Also compile before the check (default: eager, which is deterministic).",
    )
    return parser.parse_args()


def _digest(t: torch.Tensor) -> str:
    """sha256 over the raw bytes of a tensor (dtype-agnostic, incl. bfloat16)."""
    flat = t.detach().contiguous().view(-1)
    raw = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()


def _kv_pairs(past_key_values):
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(kv[0], kv[1]) for kv in past_key_values]


def main() -> int:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this check."
    torch.cuda.set_device(args.device)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
    from standalone_infer_bench import make_env_obs

    from pi05_infer import build_model
    from pi05_infer.prefix_qkv_fused import fuse_enabled, install_fused_prefix_qkv

    assert fuse_enabled(), (
        "RLINF_FUSE_PREFIX_QKV is off, so the 'on' arm would be identical to the 'off' "
        "arm and the check would trivially pass. Unset it."
    )

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
    if args.compile_mode:
        model.enable_torch_compile(mode=args.compile_mode)

    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    attn0 = layers[0].self_attn

    captured: dict = {}
    orig_build = model._build_prefix_cache  # noqa: SLF001

    def _capture(*a, **kw):
        prefix_output, pad_masks, pkv = orig_build(*a, **kw)
        captured["kv"] = [(k.clone(), v.clone()) for k, v in _kv_pairs(pkv)]
        return prefix_output, pad_masks, pkv

    model._build_prefix_cache = _capture  # noqa: SLF001

    # Layer 0's raw projection outputs, taken at the module boundary.
    proj_out: dict = {}

    def _hook(name):
        def fn(_mod, inputs, output):
            proj_out[name] = output.detach().clone()
            proj_out[name + "_in"] = inputs[0].detach().clone()

        return fn

    handles = [
        getattr(attn0, n).register_forward_hook(_hook(n))
        for n in ("q_proj", "k_proj", "v_proj")
    ]

    env_obs = make_env_obs(args)
    with torch.no_grad():
        model.predict_action_batch(env_obs)  # warmup / lazy init
        model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    kv_off = captured["kv"]
    l0_in = proj_out["q_proj_in"]
    l0_off = {n: proj_out[n] for n in ("q_proj", "k_proj", "v_proj")}
    for h in handles:
        h.remove()

    n_patched = install_fused_prefix_qkv(model)
    print(f"install_fused_prefix_qkv patched {n_patched} layers")
    assert n_patched == len(layers), (
        f"expected all {len(layers)} prefix layers to be patched, got {n_patched}"
    )

    with torch.no_grad():
        model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    kv_on = captured["kv"]

    # Direct q/k/v comparison on layer 0, from the same input the unfused arm saw.
    with torch.no_grad():
        qkv = torch.nn.functional.linear(l0_in, attn0._pi05_qkv_w)  # noqa: SLF001
        q_f, k_f, v_f = torch.split(qkv, attn0._pi05_qkv_split, dim=-1)  # noqa: SLF001
    l0_on = {"q_proj": q_f, "k_proj": k_f, "v_proj": v_f}

    print("\nlayer-0 raw projection outputs (unfused vs fused weight)")
    proj_bad = 0
    for n in ("q_proj", "k_proj", "v_proj"):
        same = torch.equal(l0_off[n].contiguous(), l0_on[n].contiguous())
        proj_bad += not same
        print(
            f"  {n:8s} {'IDENTICAL' if same else 'DIFFERS':10s} "
            f"{_digest(l0_off[n])[:16]} vs {_digest(l0_on[n].contiguous())[:16]}"
        )

    print("\nprefix KV cache, per layer")
    bad = 0
    for i, ((ka, va), (kb, vb)) in enumerate(zip(kv_off, kv_on)):
        rk, rv = torch.equal(ka, kb), torch.equal(va, vb)
        bad += (not rk) + (not rv)
        print(
            f"  layer {i:2d}  k {'SAME' if rk else 'DIFF'}  v {'SAME' if rv else 'DIFF'}"
            f"   {_digest(ka)[:12]}/{_digest(kb)[:12]}"
        )

    comb_off = hashlib.sha256()
    comb_on = hashlib.sha256()
    for (ka, va), (kb, vb) in zip(kv_off, kv_on):
        for h, t in ((comb_off, ka), (comb_off, va), (comb_on, kb), (comb_on, vb)):
            h.update(_digest(t).encode())
    print(f"\ncombined KV  off={comb_off.hexdigest()}")
    print(f"             on ={comb_on.hexdigest()}")
    ok = bad == 0 and proj_bad == 0
    print(f"\nVERDICT: {'BIT-IDENTICAL' if ok else f'{bad + proj_bad} tensors DIFFER'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
