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
r"""Do eager ``dense(cond)`` and inductor's compiled ``addmm`` agree on this shape?

``bitexact_compiled_toggles.py --disable adarms`` shows the table arm and the
per-dense arm diverging under ``max-autotune`` while agreeing in eager. The
candidate mechanism: the table is built by *eager* ``dense(cond)`` calls outside
every compiled region, while the baseline computes the same projection *inside*
the compiled expert, where inductor emits a fused Triton kernel.

This measures exactly that, on the real ``dense`` weights of all 37 adaRMS norms
and the real ``cond`` for denoise step 0. A difference is a sufficient
explanation; no difference falsifies the hypothesis and leaves the FAIL
unexplained -- which must then be reported as unexplained.

Usage::

    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_bf python tools/bitexact_adarms_dense.py --out /tmp/dense.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "bench"))


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
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--out", default=None)
    return p.parse_args()


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
    ).to(args.device).eval()

    import openpi.models.model as _model

    env_obs = make_env_obs(args)
    obs = model.obs_processor(env_obs)
    obs = model.input_transform(obs, transpose=False)
    obs = model.precision_processor(obs)
    obs = _model.Observation.from_dict(obs)
    with torch.no_grad():
        _i, _im, _lt, _lm, state = model._preprocess_observation(obs, train=False)  # noqa: SLF001

    b = state.shape[0]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    noise = torch.randn(
        (b, model.config.action_horizon, model.config.action_dim),
        device=args.device, dtype=torch.float32,
    )
    timesteps = model._get_timesteps(args.num_steps, state.device)  # noqa: SLF001
    with torch.no_grad():
        _e, _p, _a, cond = model.embed_suffix(state, noise, timesteps[0].expand(b))
    assert cond is not None, "embed_suffix returned no adaRMS cond; model is not pi05"
    print(f"cond {tuple(cond.shape)} {cond.dtype}")

    exp = model.paligemma_with_expert.gemma_expert.model
    norms = []
    for layer in exp.layers:
        norms.append(layer.input_layernorm)
        norms.append(layer.post_attention_layernorm)
    norms.append(exp.norm)
    norms = [n for n in norms if getattr(n, "dense", None) is not None]
    print(f"adaRMS dense modules: {len(norms)}  "
          f"W {tuple(norms[0].dense.weight.shape)} {norms[0].dense.weight.dtype}")

    def _lin(x, w, bias):
        return torch.nn.functional.linear(x, w, bias)

    lin_c = torch.compile(_lin, mode=args.compile_mode, fullgraph=True)

    rows, ndiff_tot, worst = [], 0, 0.0
    with torch.no_grad():
        for i, n in enumerate(norms):
            w, bs = n.dense.weight, n.dense.bias
            e = n.dense(cond)
            c = lin_c(cond, w, bs)
            torch.cuda.synchronize()
            eq = bool(torch.equal(e, c))
            d = (e.float() - c.float()).abs()
            nd = int((e != c).sum().item())
            ndiff_tot += nd
            worst = max(worst, d.max().item())
            rows.append({"i": i, "equal": eq, "max_abs": d.max().item(),
                         "ndiff": nd, "numel": int(e.numel())})
            if i < 3 or not eq:
                print(f"  norm {i:2d}  equal={eq}  max|d|={d.max().item():.3e}  "
                      f"ndiff={nd}/{e.numel()}")

    n_eq = sum(1 for r in rows if r["equal"])
    print("\neager dense(cond) vs compiled F.linear on the SAME weights/cond:")
    print(f"  bit-identical norms: {n_eq}/{len(rows)}")
    print(f"  total differing elements: {ndiff_tot}/{sum(r['numel'] for r in rows)}")
    print(f"  worst max|d|: {worst:.3e}")
    if n_eq == len(rows):
        v = ("NO DIFFERENCE -- the eager/compiled addmm split does NOT explain the "
             "table's compiled-path FAIL; the cause is unexplained.")
    else:
        v = ("DIFFERENT -- eager and inductor compute this projection with different "
             "accumulation, which is a sufficient explanation for the table arm and the "
             "per-dense arm diverging once compiled.")
    print(f"VERDICT: {v}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"verdict": v, "bit_identical": n_eq, "n_norms": len(rows),
                       "worst_max_abs": worst, "ndiff_total": ndiff_tot,
                       "compile_mode": args.compile_mode, "rows": rows,
                       "cond_shape": list(cond.shape), "cond_dtype": str(cond.dtype),
                       "gpu": torch.cuda.get_device_name(args.device),
                       "torch": torch.__version__}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
