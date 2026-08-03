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
r"""Where does a ``--dump-actions`` run stop being reproducible across processes?

The end-to-end gate compares only the final actions, so when two runs of the same
arm disagree there is no way to tell whether it entered in the prefix, the noise
draw, or the denoise loop. This runs the same seeded call and digests every
intermediate::

    obs/images, obs/state      the preprocessed model inputs (must never move)
    noise                      the initial flow-matching noise draw
    prefix/out, prefix/kv      the VLM prefix and its KV cache
    step<i>/x_in, step<i>/mean per denoise step
    actions                    what --dump-actions writes

Run N times and diff the JSON: the first stage whose digest is not constant is
where the nondeterminism lives. It also fingerprints inductor's runtime autotune
winners, so "the winners moved" and "the numbers moved" can be correlated.

Usage::

    python tools/determinism_probe.py --out /tmp/probe_r1.json
    python tools/determinism_probe.py --compare /tmp/probe_r*.json
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
)


def _sha(t) -> str:
    import torch

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


def best_config_fingerprint(cache_dir: str) -> dict:
    """Every inductor ``*.best_config`` -- the runtime Triton autotune winners.

    These are the XBLOCK/R0_BLOCK/num_warps picks for the generated pointwise and
    reduction kernels. Under ``max-autotune`` they are chosen by *benchmarking at
    first launch*, so they can move between processes; for a reduction that changes
    the accumulation split and hence the last bits of the result.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return {}
    out = {}
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".best_config"):
                p = os.path.join(root, f)
                try:
                    with open(p) as fh:
                        body = fh.read().strip()
                except OSError:
                    continue
                out[f[:16]] = body
    return out


def kernel_name_for(cache_dir: str, prefix: str) -> str:
    """Best-effort ``triton_..._kernel_name`` for a ``*.best_config`` file prefix."""
    for root, _dirs, files in os.walk(cache_dir or ""):
        for f in files:
            if f.startswith(prefix) and f.endswith(".best_config"):
                for sib in os.listdir(root):
                    if sib.endswith(".py"):
                        try:
                            with open(os.path.join(root, sib)) as fh:
                                for line in fh:
                                    if "'kernel_name':" in line:
                                        return line.split("'kernel_name': ")[1].split(
                                            ","
                                        )[0]
                        except OSError:
                            pass
                return "?"
    return "?"


def probe(args) -> dict:
    import standalone_infer_bench as B  # noqa: E402  (bench dir pushed onto sys.path)
    import torch

    if args.no_coordesc:
        # Coordinate-descent tuning re-benchmarks the neighbourhood of the cached winner
        # in *every* process (CachingAutotuner.run: "not found_by_coordesc and
        # coordinate_descent_tuning"), so a warm autotune cache does not pin the launch
        # config. Neutralise it to test whether that is what moves the numbers.
        from torch._inductor.runtime import triton_heuristics as TH

        TH.CachingAutotuner.coordinate_descent_tuning = lambda self, launcher, *a, **k: (
            launcher
        )

    if args.deterministic:
        # Collapses inductor's per-kernel reduction/pointwise autotune candidate list to a
        # single config (triton_heuristics.disable_pointwise_autotuning), so the winner
        # cannot move between processes. Costs performance; only for the numerical gate.
        torch.use_deterministic_algorithms(True, warn_only=True)

    model = B.build_model(args)
    env_obs = B.make_env_obs(args)

    # Absorb compile/autotune exactly like the --dump-actions path does.
    for _ in range(args.warmup_calls):
        with torch.no_grad():
            model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    if args.stage1:
        B.verify_stage1(model)

    rec: dict = {}
    step = {"i": 0}

    orig_pre = model._preprocess_observation
    orig_prefix = model._build_prefix_cache
    orig_noise = model.sample_noise
    orig_smvv = model.sample_mean_var_val

    def pre(*a, **k):
        out = orig_pre(*a, **k)
        images, img_masks, lang_tokens, lang_masks, state = out
        rec.setdefault("obs/images", _sha_many(images))
        rec.setdefault("obs/state", _sha(state))
        rec.setdefault("obs/lang", _sha(lang_tokens))
        return out

    def prefix(*a, **k):
        if args.freeze_prefix and os.path.exists(args.freeze_prefix):
            # Replay a prefix captured by an earlier process. Every optimisation shipped
            # so far lives in the denoise/expert path, so freezing the VLM prefix removes
            # the one stage that is known to move between processes and turns the action
            # dump into a gate on the denoise loop alone.
            blob = torch.load(args.freeze_prefix, map_location=args.device)
            out = orig_prefix(*a, **k)
            _po, ppm, pkv = out
            ppm.copy_(blob["pad"])
            for (kk, vv), (sk, sv) in zip(model._denoise_kv_pairs(pkv), blob["kv"]):
                kk.copy_(sk)
                vv.copy_(sv)
            out = (_po, ppm, pkv)
        else:
            out = orig_prefix(*a, **k)
            if args.freeze_prefix:
                _po, ppm, pkv = out
                torch.save(
                    {
                        "pad": ppm.detach().clone().cpu(),
                        "kv": [
                            (k_.detach().clone().cpu(), v_.detach().clone().cpu())
                            for k_, v_ in model._denoise_kv_pairs(pkv)
                        ],
                    },
                    args.freeze_prefix,
                )
        prefix_output, prefix_pad_masks, pkv = out
        rec.setdefault("prefix/out", _sha(prefix_output))
        rec.setdefault("prefix/pad", _sha(prefix_pad_masks))
        pairs = model._denoise_kv_pairs(pkv)
        rec.setdefault("prefix/kv", _sha_many([t for p in pairs for t in p]))
        for li, (kk, vv) in enumerate(pairs[: args.kv_layers]):
            rec.setdefault(f"prefix/kv{li}_k", _sha(kk))
            rec.setdefault(f"prefix/kv{li}_v", _sha(vv))
        return out

    noise_calls = {"n": 0}

    def noise(*a, **k):
        out = orig_noise(*a, **k)
        n = noise_calls["n"]
        noise_calls["n"] = n + 1
        if n < 2:
            rec[f"noise{n}"] = _sha(out)
        return out

    def smvv(x_t, idx, *a, **k):
        i = step["i"]
        step["i"] = i + 1
        rec[f"step{i}/x_in"] = _sha(x_t)
        out = orig_smvv(x_t, idx, *a, **k)
        rec[f"step{i}/mean"] = _sha(out[0])
        return out

    model._preprocess_observation = pre
    model._build_prefix_cache = prefix
    model.sample_noise = noise
    model.sample_mean_var_val = smvv

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        actions = model.predict_action_batch(env_obs)
    torch.cuda.synchronize()
    rec["actions"] = _sha(actions)

    payload = {
        "digests": rec,
        "meta": {
            "compile_mode": "eager" if args.no_compile else B.resolve_compile_mode(args),
            "stage1": bool(args.stage1),
            "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(args.device),
            "pid": os.getpid(),
        },
        "best_config": best_config_fingerprint(
            os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        ),
    }
    if args.save_actions:
        torch.save(actions.detach().cpu(), args.save_actions)
    return payload


def compare(paths) -> int:
    runs = []
    for p in paths:
        with open(p) as fh:
            runs.append((os.path.basename(p), json.load(fh)))
    keys = list(runs[0][1]["digests"])
    print(f"{len(runs)} runs: {', '.join(n for n, _ in runs)}")
    first_bad = None
    for k in keys:
        vals = [r["digests"].get(k) for _, r in runs]
        uniq = sorted(set(vals))
        flag = "OK " if len(uniq) == 1 else "VAR"
        if len(uniq) > 1 and first_bad is None:
            first_bad = k
        if len(uniq) > 1 or k in ("actions", "noise0", "prefix/out"):
            print(f"  {flag} {k:20s} {len(uniq)} distinct  {[v[:8] for v in vals]}")
    print(f"\nfirst diverging stage: {first_bad}")

    bc = [r.get("best_config", {}) for _, r in runs]
    if any(bc):
        allk = sorted(set().union(*[set(d) for d in bc]))
        moved = [k for k in allk if len({d.get(k) for d in bc}) > 1]
        print(
            f"\nbest_config files: {[len(d) for d in bc]} per run, "
            f"{len(moved)} differ across runs"
        )
        cd = runs[0][1]["meta"].get("cache_dir", "")
        for k in moved[:20]:
            print(f"  BCVAR {k} ({kernel_name_for(cd, k)})")
            for (n, _), d in zip(runs, bc):
                cfg = d.get(k, "")
                if isinstance(cfg, str) and cfg.startswith("{"):
                    try:
                        c = json.loads(cfg)
                        cfg = (
                            " ".join(
                                f"{a}={c[a]}"
                                for a in ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")
                                if a in c
                            )
                            + f" coordesc={c.get('found_by_coordesc')}"
                        )
                    except json.JSONDecodeError:
                        pass
                print(f"      {n:22s} {cfg}")
        # Does "same winners" imply "same numbers"?
        sig = [
            hashlib.sha256(
                json.dumps({k: d.get(k) for k in moved}, sort_keys=True).encode()
            ).hexdigest()[:8]
            for d in bc
        ]
        acts = [r["digests"].get("actions", "")[:8] for _, r in runs]
        print("\n  run                    autotune-winner-sig   actions")
        for (n, _), s, a in zip(runs, sig, acts):
            print(f"  {n:22s} {s:20s}  {a}")
        groups = {}
        for s, a in zip(sig, acts):
            groups.setdefault(s, set()).add(a)
        split = {s: v for s, v in groups.items() if len(v) > 1}
        print(
            f"  winner-signature classes: {len(groups)}; "
            f"action classes: {len(set(acts))}; "
            f"same-winners-but-different-actions: {len(split)}"
        )
    return 0 if first_bad is None else 1


def main() -> None:
    argv = sys.argv[1:]
    if "--compare" in argv:
        i = argv.index("--compare")
        sys.exit(compare(argv[i + 1 :]))

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"
        ),
    )
    import standalone_infer_bench as B

    # Reuse the bench's own argument surface so the probe builds the identical model.
    saved = sys.argv
    sys.argv = [saved[0]]
    args = B.parse_args()
    sys.argv = saved

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--save-actions", default=None)
    p.add_argument("--warmup-calls", type=int, default=2)
    p.add_argument("--kv-layers", type=int, default=2)
    p.add_argument("--stage1", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--compile-mode", default="max-autotune")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="torch.use_deterministic_algorithms(True, warn_only=True) before compile.",
    )
    p.add_argument(
        "--no-coordesc",
        action="store_true",
        help="Neutralise inductor's coordinate-descent tuning (it re-benchmarks in "
        "every process even on a warm autotune cache).",
    )
    p.add_argument(
        "--freeze-prefix",
        default=None,
        help="Path to a captured VLM prefix (pad mask + KV cache). Written if absent, "
        "replayed if present, so the denoise loop is gated on identical inputs.",
    )
    over = p.parse_args()
    for k, v in vars(over).items():
        setattr(args, k, v)

    payload = probe(args)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")
    print("actions", payload["digests"]["actions"])


if __name__ == "__main__":
    main()
