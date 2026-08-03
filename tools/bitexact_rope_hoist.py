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
"""Kernel-level bit-exactness gate for the hoisted denoise step invariants.

``RLINF_HOIST_STEP_INVARIANTS=1`` moves three tensors out of the denoise loop:

* ``position_ids`` and the 4-D attention mask -- built by the *same* eager ops as before,
  just once per predict instead of once per Euler step. Integer / bool / ``torch.where``
  arithmetic, so equality is exact by construction; checked here anyway.
* the rotary ``cos``/``sin`` table -- this one moves *across a compilation boundary*. With
  the hoist off it is produced by inductor inside the compiled expert graph
  (``triton_poi_fused__to_copy_add_cos_mean_mul_pow_rsqrt_sin_*``); with it on it is
  produced eagerly and handed to the graph as an input. That is the only place where the
  two arms can disagree in the last bit, so it is the thing worth gating.

The gate therefore compares, at the real shapes and on the real ``position_ids``:

    eager  rotary_emb(probe, position_ids)      (what the hoist computes)
    vs
    torch.compile(rotary_emb, mode=<the mode the denoise expert is compiled with>)

A bitwise match means every downstream denoise kernel sees byte-identical cos/sin, which
is the strongest evidence available without re-running the whole model -- and unlike an
end-to-end action diff it is not confounded by per-process autotune drift.

Usage:
    /opt/venv/openpi/bin/python tools/bitexact_rope_hoist.py [--prefix-len 968]
"""

import argparse
import os
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,
    )
    p.add_argument("--config-name", default="pi05_turtle")
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--prefix-len", type=int, default=968)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from pi05_infer import build_model

    model = build_model(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=3,
        noise_level=0.5,
        action_chunk=args.action_horizon,
        num_steps=10,
        train_expert_only=True,
        action_env_dim=6,
        noise_method="flow_sde",
    )
    model = model.to(args.device).eval()
    device = torch.device(args.device)
    expert = model.paligemma_with_expert.gemma_expert.model

    from pi05_infer.openpi_patched.pi0_pytorch import make_att_2d_masks

    b, m, n = args.batch_size, args.action_horizon, args.prefix_len
    prefix_pad_masks = torch.ones(b, n, dtype=torch.bool, device=device)
    state = torch.zeros(b, 32, device=device)
    x_t = torch.zeros(b, m, 32, device=device)

    hidden_dtype = expert.layers[0].self_attn.q_proj.weight.dtype
    probe = torch.zeros((), dtype=hidden_dtype, device=device)

    with torch.no_grad():
        # Reference = the per-step expressions, verbatim from get_suffix_out's
        # non-hoisted branch and GemmaModel.forward.
        s_pad, s_att, _ = model._suffix_masks(state, x_t, device)  # noqa: SLF001
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(b, s_pad.shape[1], n)
        full = torch.cat([prefix_pad_2d, make_att_2d_masks(s_pad, s_att)], dim=2)
        mask_ref = model._prepare_attention_masks_4d(full)  # noqa: SLF001
        position_ids = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(s_pad, dim=1) - 1
        )
        cos_e, sin_e = expert.rotary_emb(probe, position_ids)

        compiled = torch.compile(expert.rotary_emb.forward, mode=args.compile_mode)
        cos_c, sin_c = compiled(probe, position_ids)

    ok = True
    print(
        f"position_ids: {position_ids.dtype} {tuple(position_ids.shape)} "
        f"[{position_ids.min().item()}..{position_ids.max().item()}]"
    )
    print(
        f"cos/sin: {cos_e.dtype} {tuple(cos_e.shape)}   compile mode={args.compile_mode}"
    )
    print(f"{'tensor':>6s} {'bitwise':>8s} {'max|delta|':>12s} {'differing':>12s}")
    for name, a, c in (("cos", cos_e, cos_c), ("sin", sin_e, sin_c)):
        eq = torch.equal(a, c)
        d = (a.double() - c.double()).abs()
        ndiff = int((a != c).sum().item())
        ok &= eq
        print(
            f"{name:>6s} {str(eq):>8s} {d.max().item():12.2e} {ndiff:6d}/{a.numel():<6d}"
        )

    # The mask / position-id invariants: same expression, moved out of the loop. This only
    # guards against a transcription slip in _compute_step_invariants.
    with torch.no_grad():
        hoisted_mask, hoisted_pos, hoisted_cos, hoisted_sin = (
            model._compute_step_invariants(state, x_t, prefix_pad_masks)  # noqa: SLF001
        )
    for name, a, c in (
        ("mask", mask_ref, hoisted_mask),
        ("posid", position_ids, hoisted_pos),
        ("cos_h", cos_e, hoisted_cos),
        ("sin_h", sin_e, hoisted_sin),
    ):
        eq = torch.equal(a, c)
        ok &= eq
        print(f"{name:>6s} {str(eq):>8s} {'-':>12s} {'-':>12s}")

    print(f"\nVERDICT: {'PASS -- bitwise identical' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
