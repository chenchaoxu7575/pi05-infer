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
"""Reference arm of the A/B: drives the *RLinf* inference path in the container.

Same dummy observations, same seed, same timing harness as
``bench/standalone_infer_bench.py``, but the model comes from
``rlinf.models.embodiment.openpi.get_model``. Used to
  1. produce reference actions for the bit-exactness check, and
  2. produce a paired wall-clock number measured in the same session/thermal
     conditions as the pi05_infer arm.

This script only READS the RLinf checkout; it never writes to it.

Usage:
    /opt/venv/openpi/bin/python tools/ab_rlinf_reference.py \
        --rlinf-root /path/to/RLinf \
        --dump-actions /tmp/ref_rlinf.pt
"""

import argparse
import os
import pathlib
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bench"))
from standalone_infer_bench import _stats_ms, make_env_obs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rlinf-root",
        default=os.environ.get("RLINF_ROOT"),
        required="RLINF_ROOT" not in os.environ,
        help="Checkout of RLinf to use as the reference arm.",
    )
    p.add_argument("--model-path", default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,)
    p.add_argument("--config-name", default="pi05_turtle")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--action-chunk", type=int, default=50)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--prompt", default="Press the button with the end-effector.")
    p.add_argument("--compile-mode", default="max-autotune")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dump-actions", default=None)
    p.add_argument("--cuda-profiler", action="store_true")
    return p.parse_args()


def build_rlinf_model(args):
    sys.path.insert(0, args.rlinf_root)
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
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
        }
    )
    model = get_model(cfg).to(args.device).eval()
    if not args.no_compile:
        model.enable_torch_compile(mode=args.compile_mode)
    return model


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    print(f"gpu: {torch.cuda.get_device_name(args.device)}  torch: {torch.__version__}")
    print(f"ARM: RLINF  root={args.rlinf_root}")

    model = build_rlinf_model(args)
    env_obs = make_env_obs(args)

    if args.dump_actions:
        # Absorb compile/autotune first (see the note in standalone_infer_bench.py).
        for _ in range(2):
            with torch.no_grad():
                model.predict_action_batch(env_obs, mode="eval")
        torch.cuda.synchronize()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(env_obs, mode="eval")
        torch.save(actions.detach().cpu(), args.dump_actions)
        print(f"wrote reference actions {tuple(actions.shape)} -> {args.dump_actions}")

    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs, mode="eval")
    torch.cuda.synchronize()

    if args.cuda_profiler:
        torch.cuda.profiler.start()
    wall_ms, gpu_ms = [], []
    for i in range(args.iters):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"bench/iter{i}"):
            t0 = time.perf_counter()
            s.record()
            with torch.no_grad():
                model.predict_action_batch(env_obs, mode="eval")
            e.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1e3)
        gpu_ms.append(s.elapsed_time(e))
    if args.cuda_profiler:
        torch.cuda.profiler.stop()

    print(f"\n=== e2e predict_action_batch (RLINF arm, bs={args.batch_size}, n={args.iters}) ===")
    print(f"  cpu wall clock [ms]  {_stats_ms(wall_ms)}")
    print(f"  gpu event span [ms]  {_stats_ms(gpu_ms)}")


if __name__ == "__main__":
    main()
