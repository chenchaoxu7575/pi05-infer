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
r"""Measure the *achievable* dense bf16 tensor-core throughput of this GPU.

MFU needs a measured denominator, not an assumed peak. Compute-side twin of the
DRAM bandwidth probe.

Two traps this is built to avoid:

1. ⚠️  **Never fill operands with ``torch.empty``/``zeros``.** CUDA returns
   zeroed pages; almost no bits toggle, the power cap is never hit, the clock runs
   ~18 % high and the "peak" is a fantasy. ``--data zeros`` reproduces that on
   purpose.
2. ⚠️  **Never report a cold burst as the peak.** Runtime tracks SM clock nearly
   1:1, so 30 cold iterations and 1000 sustained ones are different machines. Both
   are reported, with the clock trace.

Usage::

    python tools/peak_bf16_gemm.py --sizes 8192,12288,16384 --backends cublas,triton --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import torch


# --------------------------------------------------------------------------- #
# clock / power sampling
# --------------------------------------------------------------------------- #
class ClockSampler:
    """Background pynvml sampler for SM clock, power and temperature.

    The GEMM loop runs for tens of seconds, so a 20 ms polling thread costs
    nothing and is the only way to tell a thermally/power-throttled sustained
    number from a boost-clock burst.
    """

    def __init__(self, device_index: int, period_s: float = 0.02) -> None:
        self.device_index = device_index
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, int, float, int]] = []
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:  # pragma: no cover - pynvml missing is not fatal
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _loop(self) -> None:
        p = self._pynvml
        while not self._stop.is_set():
            try:
                sm = p.nvmlDeviceGetClockInfo(self._handle, p.NVML_CLOCK_SM)
                w = p.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                t = p.nvmlDeviceGetTemperature(self._handle, p.NVML_TEMPERATURE_GPU)
                self.samples.append((time.perf_counter(), sm, w, t))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def start(self) -> None:
        if not self.available:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self.available or self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self.samples:
            return {}
        sm = [s[1] for s in self.samples]
        w = [s[2] for s in self.samples]
        t = [s[3] for s in self.samples]
        return {
            "n_samples": len(sm),
            "sm_mhz_mean": statistics.mean(sm),
            "sm_mhz_median": statistics.median(sm),
            "sm_mhz_min": min(sm),
            "sm_mhz_max": max(sm),
            "power_w_mean": statistics.mean(w),
            "power_w_max": max(w),
            "temp_c_max": max(t),
        }

    def window(self, t0: float, t1: float) -> dict:
        """Summarise only the samples inside [t0, t1] (perf_counter clock)."""
        sel = [s for s in self.samples if t0 <= s[0] <= t1]
        if not sel:
            return {}
        sm = [s[1] for s in sel]
        w = [s[2] for s in sel]
        return {
            "n_samples": len(sm),
            "sm_mhz_mean": statistics.mean(sm),
            "sm_mhz_median": statistics.median(sm),
            "sm_mhz_min": min(sm),
            "sm_mhz_max": max(sm),
            "power_w_mean": statistics.mean(w),
            "power_w_max": max(w),
        }


# --------------------------------------------------------------------------- #
# operand construction
# --------------------------------------------------------------------------- #
def make_operands(m: int, n: int, k: int, dev: str, data: str, linear: bool = False):
    """Build the two bf16 operands.

    ``randn`` is the only honest choice; ``zeros`` exists to demonstrate the
    clock-inflation artefact, and ``uniform`` is a second non-degenerate control
    so a result cannot be blamed on the normal distribution's tails.

    ``linear=True`` lays the second operand out as ``[N, K]`` -- the way a
    ``nn.Linear`` weight actually sits in memory -- so the benchmark reproduces
    the ``_tn_`` GEMM that the model dispatches, not a ``_nn_`` one that would
    hit different cuBLAS heuristics and different tile efficiency.
    """
    bshape = (n, k) if linear else (k, n)
    if data == "zeros":
        a = torch.zeros(m, k, device=dev, dtype=torch.bfloat16)
        b = torch.zeros(*bshape, device=dev, dtype=torch.bfloat16)
    elif data == "uniform":
        a = (torch.rand(m, k, device=dev) * 2 - 1).to(torch.bfloat16)
        b = (torch.rand(*bshape, device=dev) * 2 - 1).to(torch.bfloat16)
    else:
        a = torch.randn(m, k, device=dev, dtype=torch.float32).to(torch.bfloat16)
        b = torch.randn(*bshape, device=dev, dtype=torch.float32).to(torch.bfloat16)
    return a, b


def operand_entropy(a: torch.Tensor) -> dict:
    """Cheap sanity check that the operands are not degenerate."""
    f = a.flatten()[: 1 << 20].to(torch.float32)
    return {
        "absmean": float(f.abs().mean()),
        "nonzero_frac": float((f != 0).float().mean()),
    }


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def graphed(fn, reps: int):
    """Capture ``reps`` back-to-back invocations into one CUDA graph.

    Needed for the small prefix shapes: a 14 us GEMM launched from Python is
    dispatch-bound, and the measurement would report the launch rate rather than
    the kernel's throughput. One replay = ``reps`` GEMMs with zero launch cost.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return g.replay


def build_fn(backend: str, a: torch.Tensor, b: torch.Tensor, linear: bool = False):
    """Return a zero-argument callable that performs one A@B."""
    if backend == "cublas":
        if linear:
            return lambda: torch.nn.functional.linear(a, b)
        return lambda: torch.mm(a, b)

    # inductor-generated backends. The env var must already be set before the
    # first compile in this process (inductor reads it at autotune time), which
    # is why --backends runs one backend per process by default.
    import torch._inductor.config as icfg

    if backend == "triton":
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON"
        icfg.max_autotune_gemm_backends = "TRITON"
    elif backend == "cutlass":
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "CUTLASS"
        icfg.max_autotune_gemm_backends = "CUTLASS"
        # inductor's CUTLASS backend needs the cutlass python package + a source dir
        try:
            icfg.cuda.cutlass_max_profiling_configs = 64
        except Exception:
            pass
    elif backend == "aten_triton":
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON"
        icfg.max_autotune_gemm_backends = "ATEN,TRITON"
    else:
        raise SystemExit(f"unknown backend {backend}")

    op = torch.nn.functional.linear if linear else torch.mm
    compiled = torch.compile(
        lambda x, y: op(x, y), mode="max-autotune-no-cudagraphs", dynamic=False
    )
    return lambda: compiled(a, b)


def kernel_names(fn, n_iter: int = 3) -> list[str]:
    """Record which CUDA kernels actually ran, so 'cuBLAS' is not a guess."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    names: dict[str, int] = {}
    for e in prof.key_averages():
        if e.device_time_total > 0 and e.key not in ("cudaLaunchKernel",):
            names[e.key] = names.get(e.key, 0) + 1
    return sorted(names)


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def time_loop(fn, n_iter: int, sampler: ClockSampler) -> dict:
    """Run ``fn`` n_iter times back to back; return per-window throughput.

    One sync at the end only: at 8k-16k these GEMMs are 10-40 ms each, so launch
    overhead is <0.1 % and per-iteration syncing would only add idle gaps that
    let the clock recover -- exactly the artefact we are trying to expose.
    """
    marks: list[float] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_iter):
        fn()
        # Cheap wall-clock mark every 10 iters WITHOUT a sync: the launch queue
        # runs ahead, so these are only used to slice the clock trace, never to
        # derive throughput.
        if i % 10 == 0:
            marks.append(time.perf_counter())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {"t0": t0, "t1": t1, "elapsed_s": t1 - t0, "n_iter": n_iter}


def measure(
    backend: str,
    m: int,
    n: int,
    k: int,
    dev: str,
    data: str,
    burst_iters: int,
    sustained_s: float,
    sampler: ClockSampler,
    linear: bool = False,
    reps: int = 1,
) -> dict:
    a, b = make_operands(m, n, k, dev, data, linear)
    fn = build_fn(backend, a, b, linear)
    if reps > 1:
        fn = graphed(fn, reps)
    flop = 2.0 * m * n * k * reps

    # compile / autotune / cublas heuristic warmup, untimed
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    out = {
        "backend": backend,
        "M": m,
        "N": n,
        "K": k,
        "data": data,
        "linear": linear,
        "reps": reps,
        "flop_per_iter": flop,
        "operand_stats": operand_entropy(a),
    }

    # --- COLD BURST: let the part cool/clock back up, then a short run -------
    time.sleep(5.0)
    r = time_loop(fn, burst_iters, sampler)
    out["burst"] = {
        "n_iter": burst_iters,
        "elapsed_s": r["elapsed_s"],
        "ms_per_iter": r["elapsed_s"] / burst_iters * 1e3,
        "tflops": flop * burst_iters / r["elapsed_s"] / 1e12,
        "clocks": sampler.window(r["t0"], r["t1"]),
    }

    # --- SUSTAINED: run for sustained_s seconds, report the last third -------
    per_iter = r["elapsed_s"] / burst_iters
    n_sus = max(50, int(sustained_s / per_iter))
    r2 = time_loop(fn, n_sus, sampler)
    out["sustained"] = {
        "n_iter": n_sus,
        "elapsed_s": r2["elapsed_s"],
        "ms_per_iter": r2["elapsed_s"] / n_sus * 1e3,
        "tflops": flop * n_sus / r2["elapsed_s"] / 1e12,
        "clocks": sampler.window(r2["t0"], r2["t1"]),
    }

    # --- TAIL: a second short window immediately after, i.e. fully heat-soaked
    r3 = time_loop(fn, burst_iters, sampler)
    out["soaked"] = {
        "n_iter": burst_iters,
        "elapsed_s": r3["elapsed_s"],
        "ms_per_iter": r3["elapsed_s"] / burst_iters * 1e3,
        "tflops": flop * burst_iters / r3["elapsed_s"] / 1e12,
        "clocks": sampler.window(r3["t0"], r3["t1"]),
    }

    del a, b
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="8192,12288,16384", help="square GEMM sizes")
    p.add_argument("--shapes", default="", help="explicit MxNxK list, comma separated")
    p.add_argument("--backends", default="cublas")
    p.add_argument("--data", default="randn", choices=["randn", "uniform", "zeros"])
    p.add_argument("--burst-iters", type=int, default=30)
    p.add_argument("--sustained-s", type=float, default=30.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json", default="")
    p.add_argument("--kernel-names", action="store_true")
    p.add_argument(
        "--linear",
        action="store_true",
        help="second operand laid out [N,K] and applied via F.linear, matching how "
        "an nn.Linear weight sits in memory (dispatches the _tn_ GEMM).",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="GEMMs per CUDA-graph replay; >1 removes launch overhead for small shapes.",
    )
    args = p.parse_args()

    dev = args.device
    idx = int(dev.split(":")[1]) if ":" in dev else 0
    torch.cuda.set_device(dev)

    shapes: list[tuple[int, int, int]] = []
    if args.shapes:
        for s in args.shapes.split(","):
            m, n, k = (int(x) for x in s.lower().split("x"))
            shapes.append((m, n, k))
    else:
        for s in args.sizes.split(","):
            v = int(s)
            shapes.append((v, v, v))

    props = torch.cuda.get_device_properties(idx)
    sampler = ClockSampler(idx)
    sampler.start()

    results = {
        "device": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "data": args.data,
        "results": [],
    }
    print(f"device={props.name} SMs={props.multi_processor_count} data={args.data}")
    print(
        f"{'backend':>10s} {'shape':>18s} {'burst TF/s':>11s} {'sust TF/s':>10s} "
        f"{'soak TF/s':>10s} {'sust MHz':>9s} {'sust W':>7s}"
    )
    for backend in args.backends.split(","):
        for m, n, k in shapes:
            r = measure(
                backend,
                m,
                n,
                k,
                dev,
                args.data,
                args.burst_iters,
                args.sustained_s,
                sampler,
                args.linear,
                args.reps,
            )
            if args.kernel_names:
                a, b = make_operands(m, n, k, dev, args.data, args.linear)
                r["kernels"] = kernel_names(build_fn(backend, a, b, args.linear))
                del a, b
                torch.cuda.empty_cache()
            results["results"].append(r)
            sc = r["sustained"]["clocks"]
            print(
                f"{backend:>10s} {f'{m}x{n}x{k}':>18s} "
                f"{r['burst']['tflops']:11.1f} {r['sustained']['tflops']:10.1f} "
                f"{r['soaked']['tflops']:10.1f} "
                f"{sc.get('sm_mhz_mean', float('nan')):9.0f} "
                f"{sc.get('power_w_mean', float('nan')):7.1f}"
            )
            if "kernels" in r:
                for kn in r["kernels"]:
                    print(f"           kernel: {kn}")

    results["overall_clocks"] = sampler.stop()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
