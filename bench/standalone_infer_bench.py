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
"""Standalone pi0.5 bs=1 inference latency benchmark for ``pi05_infer``.

Adapted from ``RLinf-pi05-nsys-profile/benchmarks/pi05_infer/standalone_infer_bench.py``
(rev cbb9d2fc). Only the model-construction seam changed: ``build_model`` now calls
``pi05_infer.build_model`` instead of ``rlinf.models.embodiment.openpi.get_model``,
and ``predict_action_batch`` returns actions directly instead of
``(actions, rl_dict)``. The timing harness, the dummy observation generator and the
phase decomposition are otherwise unchanged.

Usage (inside the benchmark container):
    /opt/venv/openpi/bin/python bench/standalone_infer_bench.py \
        --model-path /workspace/rlinf_pub/models/RLinf-Pi05-Turtle-SFT

    # sync-timed phase breakdown
    ... --phases

    # under nsys (see README.md)
    nsys profile ... --capture-range=cudaProfilerApi ... --cuda-profiler
"""

import argparse
import json
import os
import statistics
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/workspace/rlinf_pub/models/RLinf-Pi05-LIBERO-SFT",
        help="Checkpoint dir (safetensors + <asset_id>/norm_stats.json).",
    )
    parser.add_argument(
        "--config-name",
        default="pi05_turtle",
        help="openpi TrainConfig name (pi05_turtle = action_horizon 50).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Flow-matching denoise steps."
    )
    parser.add_argument(
        "--action-chunk", type=int, default=50, help="Executed action chunk length."
    )
    parser.add_argument(
        "--action-dim", type=int, default=6, help="Environment action dim."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Camera views fed to the model (1 main + N-1 extra views).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input camera resolution; the openpi transform resizes to 224.",
    )
    parser.add_argument(
        "--state-dim", type=int, default=7, help="Env state dim (turtle: xyz + quat)."
    )
    parser.add_argument(
        "--prompt",
        default="Press the button with the end-effector.",
        help="Task description (tokenized/padded to 200 tokens by the transform).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (eager baseline; production runs compiled).",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="torch.compile mode for the production path.",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Enable the hand-captured denoise CUDA graph (Stage 1): one complete "
        "flow_ode step (expert forward + value + Euler + logprob) is captured and "
        "replayed per step, removing the per-step eager dispatch. Forces "
        "'max-autotune' -> 'max-autotune-no-cudagraphs', because inductor's own "
        "cudagraphs cannot be nested inside a hand-captured graph. Opt-in: the "
        "default path is unchanged.",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--phases",
        action="store_true",
        help="Also print a sync-timed per-phase decomposition of predict.",
    )
    parser.add_argument(
        "--cuda-profiler",
        action="store_true",
        help="Wrap the timed loop in torch.cuda.profiler.start()/stop() so that "
        "nsys --capture-range=cudaProfilerApi records only steady-state iters.",
    )
    parser.add_argument(
        "--dump-actions",
        default=None,
        help="Write the seeded first-call actions to this .pt file, plus a "
        "'<path>.meta.json' recording the compile mode and inductor's autotune "
        "winners. NOT a stand-alone bit-exactness gate: under 'max-autotune' the "
        "winners move between processes, so two runs of the SAME arm can differ by "
        "~3e-3. Always run an off-vs-off control (tools/bitexact_gate.sh).",
    )
    parser.add_argument(
        "--clocks-json",
        default=None,
        help="Write SM clock / power samples taken during the timed window here.",
    )
    return parser.parse_args()


# Inductor compile modes that emit their own CUDA graphs. A hand-captured graph
# cannot wrap those (the failure mode is a runtime "accessing tensor output of
# CUDAGraphs that has been overwritten"), so --stage1 rewrites them.
_CUDAGRAPH_COMPILE_MODES = {"max-autotune": "max-autotune-no-cudagraphs"}


def resolve_compile_mode(args: argparse.Namespace) -> str:
    """The compile mode actually used, after the --stage1 rewrite."""
    if not getattr(args, "stage1", False):
        return args.compile_mode
    rewritten = _CUDAGRAPH_COMPILE_MODES.get(args.compile_mode)
    if rewritten is not None:
        return rewritten
    assert "no-cudagraphs" in args.compile_mode or args.compile_mode in (
        "default",
        "reduce-overhead-no-cudagraphs",
    ), (
        f"--stage1 needs a compile mode that does not emit inductor CUDA graphs, "
        f"got --compile-mode={args.compile_mode!r}. Use "
        f"'max-autotune-no-cudagraphs' (or pass --no-compile)."
    )
    return args.compile_mode


def verify_stage1(model) -> None:
    """Fail loudly if Stage 1 silently fell back to the eager denoise loop.

    ``capture_cuda_graph`` only installs the manager; the ``torch.cuda.CUDAGraph``
    itself is captured lazily on the first eval-shaped ``sample_actions``. Both
    facts have to be checked, otherwise a shape-signature mismatch degrades to the
    eager path with no visible symptom other than the runtime.
    """
    assert model.is_cuda_graph_enabled(), (
        "--stage1 requested but is_cuda_graph_enabled() is False: "
        "capture_cuda_graph() did not install a CUDAGraphManager."
    )
    assert getattr(model, "_denoise_graph_captured", False), (
        "--stage1 requested and the manager exists, but no denoise graph was "
        "captured after warmup -- the denoise loop silently ran eager. Check "
        "_ensure_denoise_graph / the shape signature."
    )
    print(
        f"stage1 enabled: {model.is_cuda_graph_enabled()}  "
        f"denoise graph captured: {model._denoise_graph_captured}  "
        f"signature: {model._denoise_graph_spec}"
    )


def autotune_winners() -> dict:
    """Inductor's runtime Triton autotune winners for this process.

    Every generated pointwise/reduction kernel gets its launch config (XBLOCK,
    R0_BLOCK, num_warps, ...) chosen by *benchmarking at first launch*, and under
    ``max-autotune`` coordinate-descent tuning re-benchmarks the neighbourhood in
    every process -- even on a warm ``TORCHINDUCTOR_CACHE_DIR``. For a reduction
    that changes the accumulation split, so two processes running byte-identical
    code can produce different numbers. Recording the winners next to a dump makes
    that visible instead of leaving it to be discovered as a mystery diff.
    """
    import glob

    root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not root:
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        root = cache_dir()
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*.best_config"), recursive=True):
        try:
            with open(p) as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        cfg.pop("time_taken_ms", None)  # wall clock, not part of the choice
        out[os.path.basename(p)[:16]] = cfg
    return out


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Construct the model the same way the rollout worker does."""
    from pi05_infer import build_model as _build

    model = _build(
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
    mode = resolve_compile_mode(args)
    if not args.no_compile:
        if mode != args.compile_mode:
            print(
                f"--stage1: compile mode {args.compile_mode!r} -> {mode!r} "
                "(inductor cudagraphs cannot be nested in a hand-captured graph)"
            )
        model.enable_torch_compile(mode=mode)
    if getattr(args, "stage1", False):
        # Installs the manager; the graph itself is captured on the first inference.
        model.capture_cuda_graph(args.batch_size, args.batch_size)
        assert model.is_cuda_graph_enabled(), (
            "capture_cuda_graph() returned without enabling the CUDA graph manager."
        )
        print("stage1: CUDAGraphManager installed (graph captured on first predict)")
    return model


def make_env_obs(args: argparse.Namespace) -> dict:
    """Dummy observation batch in the canonical env-output format.

    Matches ``EmbodiedIOStruct.prepare_observations`` fed by the dummy Turtle2 env:
    CPU tensors, uint8 HWC images, one main camera plus (num_images - 1) extra views
    stacked on dim 1. H2D transfer happens inside predict, as in production.
    """
    g = torch.Generator().manual_seed(args.seed)
    hw = (args.image_size, args.image_size)

    def _img(*lead: int) -> torch.Tensor:
        return torch.randint(0, 256, (*lead, *hw, 3), generator=g, dtype=torch.uint8)

    num_extra = args.num_images - 1
    return {
        "main_images": _img(args.batch_size),
        "wrist_images": None,
        "extra_view_images": _img(args.batch_size, num_extra) if num_extra else None,
        "states": torch.randn(
            args.batch_size, args.state_dim, generator=g, dtype=torch.float32
        ),
        "task_descriptions": [args.prompt] * args.batch_size,
    }


def _stats_ms(samples: list[float]) -> str:
    qs = statistics.quantiles(samples, n=10)
    return (
        f"mean {statistics.mean(samples):7.2f}  p50 {statistics.median(samples):7.2f}  "
        f"p90 {qs[8]:7.2f}  min {min(samples):7.2f}  max {max(samples):7.2f}"
    )


def _read_clocks(device_index: int) -> dict:
    """SM clock (MHz), power (W) and temperature, via pynvml if available."""
    try:
        import pynvml
    except ImportError:
        return {}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return {
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "power_cap_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


def run_e2e(model, env_obs: dict, args: argparse.Namespace) -> None:
    """Time N full predict_action_batch calls: CPU wall clock + GPU span."""
    print(f"warmup x{args.warmup} (absorbs compile/autotune; may take minutes) ...")
    for _ in range(args.warmup):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    if getattr(args, "stage1", False):
        # After warmup the lazy capture must have happened; assert before timing so a
        # silent fallback to the eager denoise loop can never be reported as a result.
        verify_stage1(model)

    dev_index = torch.cuda.current_device()
    clocks = []
    if args.cuda_profiler:
        torch.cuda.profiler.start()

    wall_ms, gpu_ms = [], []
    for i in range(args.iters):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"bench/iter{i}"):
            t0 = time.perf_counter()
            start_evt.record()
            with torch.no_grad():
                model.predict_action_batch(env_obs)
            end_evt.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1e3)
        gpu_ms.append(start_evt.elapsed_time(end_evt))
        if args.clocks_json:
            clocks.append(_read_clocks(dev_index))

    if args.cuda_profiler:
        torch.cuda.profiler.stop()

    print(f"\n=== e2e predict_action_batch (bs={args.batch_size}, n={args.iters}) ===")
    print(f"  cpu wall clock [ms]  {_stats_ms(wall_ms)}")
    # GPU span (first->last GPU work), comparable to nsys projection, not busy time.
    print(f"  gpu event span [ms]  {_stats_ms(gpu_ms)}")

    if args.clocks_json:
        sm = [c["sm_clock_mhz"] for c in clocks if "sm_clock_mhz" in c]
        pw = [c["power_w"] for c in clocks if "power_w" in c]
        summary = {
            "wall_ms": wall_ms,
            "gpu_ms": gpu_ms,
            "clock_samples": clocks,
            "sm_clock_mhz_mean": statistics.mean(sm) if sm else None,
            "sm_clock_mhz_min": min(sm) if sm else None,
            "sm_clock_mhz_max": max(sm) if sm else None,
            "power_w_mean": statistics.mean(pw) if pw else None,
        }
        with open(args.clocks_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(
            f"  sm clock [MHz]       mean {summary['sm_clock_mhz_mean']}  "
            f"min {summary['sm_clock_mhz_min']}  max {summary['sm_clock_mhz_max']}  "
            f"power {summary['power_w_mean']:.1f} W"
            if sm
            else "  (pynvml unavailable; no clock samples)"
        )


def run_phases(model, env_obs: dict, args: argparse.Namespace) -> None:
    """Sync-timed decomposition of one predict call into its NVTX phases."""
    from openpi.models import model as _model

    if getattr(args, "stage1", False):
        print(
            "\nWARNING: --phases drives sample_mean_var_val directly, which bypasses "
            "the captured denoise graph. The denoise/loop row below is the EAGER "
            "cost, not the Stage-1 cost; use run_e2e for the Stage-1 number."
        )

    def timed(name: str, fn):
        # Untimed warmup call: invoking the internals directly (instead of via
        # predict_action_batch) can trigger one dynamo recompile on the first
        # call, which must not land in the timed window.
        out = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = fn()
        torch.cuda.synchronize()
        acc[name] = (time.perf_counter() - t0) / args.iters * 1e3
        return out

    acc: dict[str, float] = {}

    # Capture the prefix-phase inputs/outputs from one genuine predict so the
    # timed GPU phases see bit-identical tensors (shape, dtype, and strides).
    captured: dict[str, tuple] = {}
    orig_build_prefix_cache = model._build_prefix_cache

    def _capturing_build_prefix_cache(images, img_masks, lang_tokens, lang_masks):
        out = orig_build_prefix_cache(images, img_masks, lang_tokens, lang_masks)
        captured["prefix_args"] = (images, img_masks, lang_tokens, lang_masks)
        captured["prefix_out"] = out
        return out

    model._build_prefix_cache = _capturing_build_prefix_cache
    try:
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    finally:
        model._build_prefix_cache = orig_build_prefix_cache
    assert "prefix_args" in captured, (
        "predict_action_batch did not call _build_prefix_cache; "
        "the phase decomposition no longer matches the model code."
    )
    images, img_masks, lang_tokens, lang_masks = captured["prefix_args"]
    _, prefix_pad_masks, past_key_values = captured["prefix_out"]

    with torch.no_grad():
        # CPU-side phases (memory-format independent, re-derived per call).
        to_process = timed("predict/obs_processor", lambda: model.obs_processor(env_obs))
        transformed = timed(
            "predict/input_transform",
            lambda: model.input_transform(to_process, transpose=False),
        )
        observation = timed(
            "predict/precision+from_dict",
            lambda: _model.Observation.from_dict(
                model.precision_processor(dict(transformed))
            ),
        )
        preprocessed = timed(
            "denoise/preprocess",
            lambda: model._preprocess_observation(observation, train=False),
        )
        state = preprocessed[4]
        # GPU phases, replayed with the captured production tensors.
        timed(
            "prefix (embed+mask+vlm_forward)",
            lambda: orig_build_prefix_cache(images, img_masks, lang_tokens, lang_masks),
        )

        def denoise_loop():
            x_t = model.sample_noise(
                (args.batch_size, model.config.action_horizon, model.config.action_dim),
                args.device,
            )
            for idx in range(args.num_steps):
                x_t_mean, x_t_std, _value_t, _v_t = model.sample_mean_var_val(
                    x_t, idx, state, prefix_pad_masks, past_key_values, args.num_steps
                )
                x_t = x_t_mean + model.sample_noise(x_t.shape, args.device) * x_t_std
                model.get_logprob_norm(x_t, x_t_mean, x_t_std)
            return x_t

        actions = timed(f"denoise/loop (x{args.num_steps} steps)", denoise_loop)
        timed(
            "predict/output_transform",
            lambda: model.output_transform(
                {"actions": actions, "state": observation.state}
            ),
        )

    total = sum(acc.values())
    print(f"\n=== per-phase wall clock (bs={args.batch_size}, n={args.iters}) ===")
    for name, ms in acc.items():
        print(f"  {name:<34} {ms:8.2f} ms  ({100 * ms / total:4.1f}%)")
    print(f"  {'SUM':<34} {total:8.2f} ms")


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required for this benchmark."
    torch.cuda.set_device(args.device)

    print(f"gpu: {torch.cuda.get_device_name(args.device)}")
    print(f"torch: {torch.__version__}")
    compile_desc = "eager" if args.no_compile else resolve_compile_mode(args)
    print(
        f"config: {args.config_name} bs={args.batch_size} num_steps={args.num_steps} "
        f"chunk={args.action_chunk} images={args.num_images}x{args.image_size} "
        f"compile={compile_desc} stage1={bool(args.stage1)}"
    )

    model = build_model(args)
    env_obs = make_env_obs(args)

    if args.dump_actions:
        # Absorb compile/autotune first: inductor's autotuning draws random tensors,
        # so the seeded call below must happen with compilation already done.
        for _ in range(2):
            with torch.no_grad():
                model.predict_action_batch(env_obs)
        torch.cuda.synchronize()
        if args.stage1:
            verify_stage1(model)
        # Deterministic single call for the numerical A/B: fix the CUDA RNG so the
        # initial flow-matching noise draw is reproducible across processes.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            ref = model.predict_action_batch(env_obs)
        torch.save(ref.detach().cpu(), args.dump_actions)
        meta = {
            "compile_mode": compile_desc,
            "stage1": bool(args.stage1),
            "seed": args.seed,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(args.device),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
            "autotune_winners": autotune_winners(),
        }
        with open(args.dump_actions + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        print(f"wrote reference actions {tuple(ref.shape)} -> {args.dump_actions}")
        print(
            f"wrote {args.dump_actions}.meta.json "
            f"({len(meta['autotune_winners'])} autotune winners)"
        )

    run_e2e(model, env_obs, args)
    if args.phases:
        run_phases(model, env_obs, args)


if __name__ == "__main__":
    main()
