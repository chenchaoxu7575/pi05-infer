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
"""In-process bs=1 latency benchmark for SGLang v0.5.16's pi0.5 VLA pipeline.

Why in-process: the numbers published in sgl-project/sglang#30633 (300.70 ms ->
81.46 ms) come from the *serving* path and therefore include HTTP, msgpack and
scheduler overhead. Our own baseline (``bench/standalone_infer_bench.py``) is a
plain in-process ``predict_action_batch`` wall clock. To compare, this script
drives ``Pi05PolicyModel`` directly -- ``encode_prefix`` + ``sample_actions`` --
so both sides measure the same thing.

Config is pinned to the RLinf pi0.5 reference point: 3 cameras at 224x224
(768 image tokens) + 200 language tokens = **968 prefix tokens**, action chunk
50, K = 10 denoise steps, batch 1, bf16, no quantization.

Requires the isolated SGLang venv (torch 2.11 / cu130); it must NOT be run with
the openpi venv interpreter.

Usage:
    /path/to/sglang-venv/bin/python tools/sglang_pi05_bench.py \
        --model-path /path/to/RLinf-Pi05-LIBERO-SFT \
        --warmup 8 --iters 30 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,
        help="openpi-layout checkpoint dir (model.safetensors).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Raw camera resolution; resized to the model's 224 by preprocess.",
    )
    parser.add_argument(
        "--token-len",
        type=int,
        default=200,
        help="Language tokens with mask=1. 200 -> 968-token prefix (matches ours). "
        "SGLang trims trailing padding, so a shorter real prompt shortens the "
        "prefix and makes the comparison invalid.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable VLADenoiseGraphRunner (their eager denoise baseline).",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Force an SGLang attention backend (fa, fa2, torch_sdpa). Default: "
        "let their selector choose.",
    )
    parser.add_argument("--json", default=None, help="Write raw samples here.")
    return parser.parse_args()


def _stats(samples: list[float]) -> dict:
    qs = statistics.quantiles(samples, n=20) if len(samples) >= 20 else None
    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "sd": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p5": qs[0] if qs else min(samples),
        "p95": qs[18] if qs else max(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _fmt(name: str, s: dict) -> str:
    return (
        f"  {name:<34} mean {s['mean']:7.2f}  median {s['median']:7.2f}  "
        f"sd {s['sd']:5.3f}  min {s['min']:7.2f}  max {s['max']:7.2f}"
    )


def read_clocks(device_index: int) -> dict:
    try:
        import pynvml
    except ImportError:
        return {}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return {
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


def build_raw_images(args: argparse.Namespace) -> list[torch.Tensor]:
    """uint8 HWC CPU camera frames, exactly the format our own bench starts from."""
    g = torch.Generator().manual_seed(args.seed)
    hw = (args.image_size, args.image_size)
    return [
        torch.randint(0, 256, (*hw, 3), generator=g, dtype=torch.uint8)
        for _ in range(args.num_images)
    ]


def main() -> None:
    args = parse_args()

    from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
    from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import Pi05PolicyModel
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (  # noqa: E501
        _resize_with_pad_image_tensor,
        _tensor_from_image,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        set_global_server_args,
    )
    from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch

    # The only thing the pi0.5 model path reads out of the process-global
    # sgl_diffusion args is ``attention_backend`` (selector.py:167); the full
    # ServerArgs constructor additionally wants a diffusers-style model registry
    # entry, which an openpi checkpoint directory does not have. Install a stub
    # carrying exactly the CLI default (``attention_backend = None``) so their own
    # auto-selection runs, unchanged.
    set_global_server_args(
        SimpleNamespace(
            attention_backend=args.attention_backend,
            attention_backend_config=None,
        )
    )

    config = Pi05PipelineConfig()
    config.action_horizon = args.action_chunk
    config.n_action_steps = args.action_chunk
    config.default_num_inference_steps = args.num_steps
    config.max_token_len = args.token_len
    config.enable_action_cuda_graph = not args.no_cuda_graph
    config.enable_global_prefix_cache = False

    print(
        f"sglang pi0.5: image {config.image_size} x {args.num_images} cameras, "
        f"tokens {args.token_len}, chunk {config.action_horizon}, "
        f"K={args.num_steps}, dtype {config.materialize_dtype}, "
        f"cuda_graph={config.enable_action_cuda_graph}"
    )

    t0 = time.perf_counter()
    model = Pi05PolicyModel.from_pretrained(args.model_path, config)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    device = model.device
    camera_order = tuple(config.image_keys[: args.num_images])
    raw_images = build_raw_images(args)

    # Deterministic token ids in the PaliGemma vocab. Token *values* do not change
    # any GEMM shape, so they do not affect timing; the mask is all-ones so the
    # prefix stays at 768 + token_len and full attention (and therefore the CUDA
    # graph) is eligible, exactly as in our own configuration.
    g = torch.Generator().manual_seed(args.seed)
    tokens = torch.randint(
        1000, 200_000, (args.batch_size, args.token_len), generator=g, dtype=torch.long
    )
    token_masks = torch.ones(args.batch_size, args.token_len, dtype=torch.bool)
    fixed_noise = torch.randn(
        args.batch_size,
        config.action_horizon,
        config.action_dim,
        generator=g,
        dtype=torch.float32,
    ).to(device)

    def preprocess() -> VLAObservationBatch:
        images, image_masks = {}, {}
        for key, raw in zip(camera_order, raw_images, strict=True):
            tensor = _tensor_from_image(raw)
            tensor = _resize_with_pad_image_tensor(tensor, config.image_size)
            tensor = tensor * 2.0 - 1.0
            images[key] = tensor.unsqueeze(0)
            image_masks[key] = torch.tensor([True], dtype=torch.bool)
        return VLAObservationBatch(
            prompt=["bench"],
            images=images,
            image_masks=image_masks,
            state=None,
            noise=None,
            tokens=tokens,
            token_masks=token_masks,
            batch_size=args.batch_size,
            metadata={"camera_order": camera_order},
        )

    obs = preprocess()

    def one_call():
        ctx = model.encode_prefix(obs)
        return ctx, model.sample_actions(
            obs, ctx, noise=fixed_noise, num_steps=args.num_steps
        )

    print(f"warmup x{args.warmup} (absorbs CUDA graph capture) ...")
    for _ in range(args.warmup):
        ctx, actions = one_call()
    torch.cuda.synchronize()
    print(
        f"  prefix_len = {ctx.prefix_len}  full_attention = "
        f"{ctx.layout.get('full_attention')}  actions {tuple(actions.shape)}"
    )
    captured = list(model.graph_runner._captured.keys())
    print(f"  denoise CUDA graphs captured: {len(captured)} {captured}")
    assert not config.enable_action_cuda_graph or captured, (
        "CUDA graph requested but nothing was captured -- the denoise loop ran "
        "eager and the number below would not be their optimized path."
    )

    dev_index = torch.cuda.current_device()
    pre_ms, prefix_ms, denoise_ms, total_ms, clocks = [], [], [], [], []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t_a = time.perf_counter()
        obs_i = preprocess()
        t_b = time.perf_counter()
        ctx = model.encode_prefix(obs_i)
        torch.cuda.synchronize()
        t_c = time.perf_counter()
        model.sample_actions(obs_i, ctx, noise=fixed_noise, num_steps=args.num_steps)
        torch.cuda.synchronize()
        t_d = time.perf_counter()
        pre_ms.append((t_b - t_a) * 1e3)
        prefix_ms.append((t_c - t_b) * 1e3)
        denoise_ms.append((t_d - t_c) * 1e3)
        total_ms.append((t_d - t_a) * 1e3)
        clocks.append(read_clocks(dev_index))

    print(
        f"\n=== sglang v0.5.16 pi0.5, in-process, bs={args.batch_size}, "
        f"n={args.iters} [ms] ==="
    )
    result = {
        "preprocess (CPU resize+normalize)": _stats(pre_ms),
        "prefix (encode_prefix)": _stats(prefix_ms),
        f"denoise (sample_actions x{args.num_steps})": _stats(denoise_ms),
        "TOTAL": _stats(total_ms),
    }
    for name, s in result.items():
        print(_fmt(name, s))
    print(
        f"  denoise per step: "
        f"{statistics.median(denoise_ms) / args.num_steps * 1e3:.1f} us"
    )

    sm = [c["sm_clock_mhz"] for c in clocks if "sm_clock_mhz" in c]
    pw = [c["power_w"] for c in clocks if "power_w" in c]
    if sm:
        print(
            f"  sm clock [MHz] mean {statistics.mean(sm):.0f} "
            f"min {min(sm)} max {max(sm)}   power {statistics.mean(pw):.1f} W"
        )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "args": vars(args),
                    "prefix_len": ctx.prefix_len,
                    "full_attention": ctx.layout.get("full_attention"),
                    "cuda_graphs": [str(k) for k in captured],
                    "summary": result,
                    "raw": {
                        "preprocess_ms": pre_ms,
                        "prefix_ms": prefix_ms,
                        "denoise_ms": denoise_ms,
                        "total_ms": total_ms,
                    },
                    "clocks": clocks,
                },
                fh,
                indent=2,
            )
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
