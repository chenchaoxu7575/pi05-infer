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
"""Clock-resolved timing of the pi0.5 prefix phase, isolated and in situ.

Motivation: this part is power-limited, not clock-limited. A back-to-back bf16
GEMM pins the 300 W wall and the SM clock collapses to ~1.8 GHz; the full pi0.5
predict loop -- which is roughly half memory-bound denoise -- holds ~2.3 GHz at
the *same* 294 W. So "achievable TFLOP/s" is not one number, it is a function of
the clock the workload can hold, and the MFU denominator for the prefix depends
on which regime the prefix actually runs in.

This probe measures three loops with one instrument, so the clocks are
comparable:

  * ``predict``  -- full ``predict_action_batch`` (the shipped workload)
  * ``prefix``   -- ``_build_prefix_cache`` only, back to back
  * ``vision``   -- the SigLIP tower only, back to back (``embed_image``)

Usage:
    /opt/venv/openpi/bin/python tools/prefix_sol_probe.py \
        --model-path .../RLinf-Pi05-LIBERO-SFT --config-name pi05_turtle \
        --stage1 --seconds 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT + "/bench")
sys.path.insert(0, _ROOT + "/tools")
from peak_bf16_gemm import ClockSampler  # noqa: E402
from standalone_infer_bench import (  # noqa: E402
    build_model,
    make_env_obs,
    verify_stage1,
)


def timed_loop(fn, seconds: float, sampler: ClockSampler, min_iters: int = 20) -> dict:
    """Run ``fn`` for ~``seconds`` seconds; return wall time and the clock window."""
    fn()
    torch.cuda.synchronize()
    # calibrate
    t0 = time.perf_counter()
    for _ in range(min_iters):
        fn()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / min_iters
    n = max(min_iters, int(seconds / per))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {
        "n_iter": n,
        "ms_per_iter": (t1 - t0) / n * 1e3,
        "elapsed_s": t1 - t0,
        "clocks": sampler.window(t0, t1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,
    )
    p.add_argument("--config-name", default="pi05_turtle")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--action-chunk", type=int, default=50)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--compile-mode", default="max-autotune")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--prompt", default="Press the button with the end-effector.")
    p.add_argument("--stage1", action="store_true")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--json", default="")
    args = p.parse_args()

    torch.cuda.set_device(args.device)
    model = build_model(args)
    env_obs = make_env_obs(args)

    with torch.no_grad():
        for _ in range(args.warmup):
            model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    if args.stage1:
        verify_stage1(model)

    # Capture the production prefix inputs so the isolated loops see tensors that
    # are bit-identical in shape, dtype and stride to the shipped path.
    captured: dict = {}
    orig = model._build_prefix_cache

    def _cap(images, img_masks, lang_tokens, lang_masks):
        out = orig(images, img_masks, lang_tokens, lang_masks)
        captured["args"] = (images, img_masks, lang_tokens, lang_masks)
        return out

    model._build_prefix_cache = _cap
    try:
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    finally:
        model._build_prefix_cache = orig
    images, img_masks, lang_tokens, lang_masks = captured["args"]
    stacked = torch.cat(images, dim=0) if len(images) > 1 else images[0]

    idx = int(args.device.split(":")[1]) if ":" in args.device else 0
    sampler = ClockSampler(idx)
    sampler.start()

    out = {"config": vars(args), "loops": {}}
    with torch.no_grad():
        loops = {
            "predict": lambda: model.predict_action_batch(env_obs),
            "prefix": lambda: orig(images, img_masks, lang_tokens, lang_masks),
            "vision_siglip": lambda: model.paligemma_with_expert.embed_image(stacked),
        }
        for name, fn in loops.items():
            # 3 s of idle between loops so each one starts from the same thermal
            # state rather than inheriting the previous loop's clock.
            time.sleep(3.0)
            r = timed_loop(fn, args.seconds, sampler)
            out["loops"][name] = r
            c = r["clocks"]
            print(
                f"{name:>14s}  n={r['n_iter']:6d}  {r['ms_per_iter']:8.3f} ms/iter  "
                f"clk {c.get('sm_mhz_mean', float('nan')):.0f} MHz "
                f"(min {c.get('sm_mhz_min')} max {c.get('sm_mhz_max')})  "
                f"{c.get('power_w_mean', float('nan')):.1f} W"
            )

    out["overall_clocks"] = sampler.stop()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
