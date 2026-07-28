# Extraction notes

Every deviation from a straight copy, every coupling that remains, and everything
that could not be cleanly separated. Read this before changing anything in
`pi05_infer/`.

Source of truth for all extractions: `RLinf-pi05-nsys-profile` at rev **`cbb9d2fc`**
(`perf(pi0.5): wire denoise bs=1 optimizations into the inference path`). Verified
byte-identical to the checkout deployed in the `pi05bench` container.

---

## 0. The environment claim in the task brief was wrong

The brief said `/workspace/rlinf_pub/pi05-infer/` is "visible from BOTH the analysis
box and inside the container". It is not — the two `/workspace/rlinf_pub` trees are on
different machines:

| namespace | real path |
|---|---|
| analysis box `h20-8` | `/workspace/rlinf_pub/pi05-infer` (this is where the git repo lives) |
| GPU box `b4696b4-lcedt`, host | `/home/chenchaox/project/RLinf_pi0.5_inference/pi05-infer` |
| GPU box, inside `pi05bench` | `/workspace/rlinf_pub/pi05-infer` (bind mount of the above) |

The repo path matches the brief inside the container, which is what the code needs.
The two copies are kept in sync by `tar | ssh`; the git history lives on the analysis
box and is mirrored to the GPU box. `rsync` is not installed on the analysis box.

Related: the GPU box's copy of `RLinf-pi05-nsys-profile` is **not a git repository** —
it is an unpacked tree. Its `openpi_action_model.py` is byte-identical
(md5 `86e98eeb580dc7a74b68db3fab1e3866`) to the analysis box's `cbb9d2fc`, so the two
are the same code; but you cannot run `git` commands against the container copy.
The analysis box's working tree additionally carries uncommitted WIP
(`denoise_static_masks` + `max_token_len=48`, +49 lines) which is **not** in the
container and therefore **not** part of the measured baseline or of this extraction.

---

## 1. Stage 0 — rescue (commit 1)

Four files were copied verbatim (`docker cp`) out of the container's site-packages,
where they were untracked by any repository and one `install.sh` re-run away from
being destroyed:

| repo path | container source | lines | md5 |
|---|---|--:|---|
| `pi05_infer/gemma/modeling_gemma.py` | `transformers/models/gemma/modeling_gemma.py` | 1107 | `5beeb6cb…` |
| `pi05_infer/gemma/rlinf_fused_denoise.py` | `transformers/models/gemma/rlinf_fused_denoise.py` | 552 | `dbbf4c44…` |
| `pi05_infer/openpi_patched/pi0_pytorch.py` | `openpi/models_pytorch/pi0_pytorch.py` | 480 | `91a4a79c…` |
| `pi05_infer/openpi_patched/gemma_pytorch.py` | `openpi/models_pytorch/gemma_pytorch.py` | 285 | `ea915248…` |

**These two directories are not Python packages** (no `__init__.py`) and are **not
imported by anything**. They are rescue copies only. `pi05_infer` at Stage 1 still
resolves `transformers.models.gemma` and `openpi.models_pytorch.pi0_pytorch` from the
container's site-packages — i.e. from the same already-patched files that produced the
measured numbers. Making the package use its own vendored copies is **Stage 2 of the
plan and has NOT been done**; see §6.

---

## 2. What was dropped from `openpi_action_model.py` (1824 lines → `engine.py`)

Dropped wholesale (RL only):

- `sft_forward`, `prepare_dagger_sft_batch`, `preprocess_for_train`
- `default_forward`, `nft_forward`, `forward` dispatcher over `ForwardType`
- `get_log_prob_value`, `get_value_from_vlm`, `gaussian_entropy`,
  `_compute_value_from_suffix`, the `ValueHead` module and every `add_value_head` /
  `value_after_vlm` / `chunk_critic_input` / `detach_critic_input` branch
- `ExploreNoiseNet` / `noise_head` and the `flow_noise` sampler
- the `flow_sde` and `flow_cps` samplers
- DSRL: `use_dsrl` config group, `sac_forward`, `sac_q_forward`, the five DSRL
  encoder/head modules, `_preprocess_dsrl_images`, `_preprocess_states`, and the DSRL
  branch of `predict_action_batch` and of `freeze_vlm`
- NFT: `is_nft`, `_init_nft_state`, `_update_nft_state`, and the `nft_*` outputs
- `set_global_step` / `self.global_step` and the noise-annealing schedule
  (`noise_anneal`, `noise_params`)
- `joint_logprob`, `safe_get_logprob`, `double_layer`, `ignore_last`
- from `sample_actions`: the `chains` list and its `torch.stack`, the `denoise_inds`
  tensor and the `random.randint` train-mode branch, the `log_probs` list / stack /
  gather, the `values` list / stack, `mode="train"`, `compute_values`
- from `predict_action_batch`: the whole `forward_inputs` dict (`chains`,
  `denoise_inds`, `tokenized_prompt*`, `action`, `model_action`, the cloned obs) and
  the `(actions, result)` return shape. `predict_action_batch` now returns just
  `actions`.

Config: `OpenPi0Config` (34 fields) → `OpenPi0InferConfig` (8 fields). Retained
fields keep their original names and defaults.

### 2.1 DELIBERATE EXCEPTIONS — RL code that was NOT dropped

Three statements are RL instrumentation but sit **inside one denoise step**, i.e.
inside the region that the "238 kernels/step" regression gate measures and inside the
body captured by the Stage-1 CUDA graph. Deleting them would change the kernel count
and invalidate the A/B against the container. They are computed and the results
discarded:

1. `get_logprob_norm(x_t, x_t_mean, x_t_std)` in the eager denoise loop, and inside
   `_capture_denoise_step._step`.
2. `value_t = torch.zeros((bsize), device=device)` in `sample_mean_var_val` — in the
   original this is the `else` branch of the value-head `if`, which is always taken
   when `add_value_head=False` (the inference config). Kept as an unconditional
   statement, producing the identical device tensor.
3. `self._get_noise_level(device, dtype)` in `sample_mean_var_val` — its return value
   is unused on the flow_ode path, but the call is one tiny device-tensor construction
   per step that is present in the measured baseline.

If you ever want these gone, delete them and re-measure kernels/step and e2e; do not
delete them silently.

## 3. Line-level changes other than deletions

Beyond import rewrites, logging and dead-RL-code deletion, the following changed:

| where | change | why |
|---|---|---|
| `engine.py` `sample_mean_var_val` | the `if/elif` sampler chain collapsed to the `flow_ode` body | the other three samplers were deleted; flow_ode is the only reachable branch in eval (`denoise_inds == -1`) |
| `engine.py` `sample_mean_var_val` | value-head `if` collapsed to its `else` branch | `add_value_head=False` on the inference path |
| `engine.py` `sample_actions` | the per-step `if idx == denoise_inds[0][idx]` sampler selection removed | always `flow_ode` in eval |
| `engine.py` `sample_actions` / graph helpers | `compute_values` and `collect_nft_state` removed from signatures and from `_denoise_graph_signature` | the values they could take on the inference path are constant |
| `engine.py` `_get_noise_level` | `sample_method` parameter and the annealing branch removed | annealing is train-only; the flow_ode early-return is unreachable because the only call site never passes `sample_method` |
| `engine.py` `__init__` | `self.logger = get_logger()` → module `logging.getLogger(__name__)` | brief |
| `engine.py` | `from rlinf.utils.utils import nvtx_range` → `pi05_infer._vendored.nvtx` | `rlinf.utils.utils` imports `rlinf.scheduler.Worker` (Ray) at module scope |
| `engine.py` | `PI0Pytorch, BasePolicy` base classes kept; `ForwardType` no longer imported | no dispatcher |
| `builder.py` | `get_model(cfg: DictConfig)` → `build_model(**kwargs)` with a thin `get_model(cfg)` shim | removes the omegaconf dependency from the core path without breaking the existing call shape |
| `builder.py` | `repack_transforms = transforms.Group()` and `default_prompt = None` inlined | both were dead locals in the original (`*Group().inputs` is empty) |
| `builder.py` | `norm_stats = None; if norm_stats is None:` unwrapped | dead conditional in the original |
| `bench/standalone_infer_bench.py` | `predict_action_batch(env_obs, mode="eval")` → `predict_action_batch(env_obs)`; result unpacking removed | new return shape |
| `bench/standalone_infer_bench.py` | added `--dump-actions` and `--clocks-json` | verification support; both no-ops unless passed |
| `_vendored/nvtx.py` | the two fallback warnings use `logging` instead of `rlinf.utils.logging.get_logger` | breaking the RLinf import |
| `_vendored/base_policy.py` | one import line rewritten to `pi05_infer._vendored.cuda_graph` | verbatim otherwise |
| `dataconfig/{turtle,libero}_dataconfig.py` | one import line each rewritten to `pi05_infer.dataconfig.policies` | verbatim otherwise |
| `tools/bitgate.py` | the hard-coded gemma directory became `sys.argv[1]` with the original as default | so the rescued copy can be gated too |
| `tools/ksum.py` | none | verbatim from `claude_mem/pi05_rollout_forward/kernel_fusion/scripts/ksum.py` |

`tools/prof.sh`, `tools/stream_summary.py`, `tools/denoise_kernels.py` and
`tools/ab_rlinf_reference.py` are new (written for this extraction's verification);
`prof.sh` is adapted from `kernel_fusion/scripts/prof.sh`.

`_vendored/cuda_graph.py`, `dataconfig/policies/{turtle,libero}_policy.py` and
`dataconfig/policies/__init__.py` are byte-identical to their RLinf originals.

## 4. `dataconfig` — minimal subset only

`pi05_infer/dataconfig/__init__.py` keeps 3 of the 16 RLinf `TrainConfig`s
(`pi0_libero`, `pi05_libero`, `pi05_turtle`) and their two data configs / policies.
The other 13 (maniskill, metaworld, calvin, robocasa, robotwin_aloha, franka,
franka_co_training, dual_franka_tcp_rot6d, polaris, isaaclab, gsenv, behavior,
realworld) are **not** vendored. `get_openpi_config`, `_override_with_model_path`
and `_override_with_data_kwargs` are verbatim.

RLinf's own `dataconfig/` is untouched — it is shared by `openpi_cfg` and three
workers and cannot be moved. This is a deliberate duplicate; the two will drift.
`rlinf/models/embodiment/openpi/transforms/` (which needs `rlinf.utils.rot6d`) is not
reachable from the two retained data configs and was not vendored.

## 5. Couplings that remain

1. **openpi.** `engine.py` imports `openpi.transforms`, `openpi.models.model`,
   `openpi.models.pi0_config.Pi0Config` and subclasses
   `openpi.models_pytorch.pi0_pytorch.PI0Pytorch`. This is by design (the plan keeps
   openpi installed for `Observation`, transforms and checkpoint loading).
2. **The patched site-packages.** The measured performance depends on the container's
   *modified* `transformers/models/gemma/modeling_gemma.py` (+245 lines) and on the
   modified `openpi/models_pytorch/pi0_pytorch.py`. `pi05_infer` does not yet supply
   these itself — it inherits whatever the container has. On a pristine openpi install
   the package will import and run but will be slower and `build_adarms_stack` /
   `build_qkv_fused` / `prime_kv_static` will be missing (`enable_torch_compile` calls
   the first two **without** a `hasattr` guard, so it would raise).
3. **`RLINF_DISABLE_OPENPI_TYPECHECK`.** The env-var gate that no-ops openpi's
   `@at.typecheck` is copied as-is, including its `RLINF_` prefix, so a single env var
   still controls both codebases. Default off in both, so both arms of the A/B pay the
   same ~3.2 ms.
4. **`self.sample_actions = sample_actions_func` in `__init__`.** The rebind that
   stops `PI0Pytorch` from polymorphically calling the subclass method is preserved
   verbatim. Do not "clean it up".
5. **`tools/ab_rlinf_reference.py`** imports RLinf (read-only) — it exists purely to
   run the reference arm of the A/B and is not part of the package.

## 6. Things measurement could not settle cleanly

- **"238 kernels/step".** With `tools/denoise_kernels.py` both arms measure **234.90**.
  The tool's window runs from the first to the last stream-157 kernel of a predict, so
  the trailing eager glue of the final denoise step falls outside it; extending the
  window past that point picks up the next predict's prefix instead. Since the RLinf
  arm — the code the 238 figure was recorded against — also measures 234.90, the gap
  is a definition difference in the counting method, not a regression. If you need the
  exact historical number back, find the script that produced it; it was not in
  `claude_mem/pi05_rollout_forward/kernel_fusion/scripts/`.
- **"1236 µs/step" denoise.** Under nsys with `--gpu-metrics-devices` the measured GPU
  busy time is 1135.8 µs/step on stream 157 alone and 1613 µs/step summing streams 157
  and 158 (which overlap). Neither reproduces 1236 µs directly; both are identical
  between the two arms to within 0.07 %, which is the property that matters here.
- **Prefix busy time** is 0.39 % higher in arm B (23863.9 vs 23771.8 µs/predict) on a
  *strictly smaller* kernel set with an identical kernel-category table. This is
  run-to-run noise, not a regression, but it was a single paired run — if you touch the
  prefix, re-measure with more repetitions.
- **Only one paired A/B run** was done per arm (30 timed iterations each), and the
  reference arm ran first, i.e. on a colder card. The 0.52 ms gap is consistent with
  the 6 removed bookkeeping kernels but is not resolved beyond the ±0.7 ms rebuild
  variance by a single pair.

## 7. Not done (out of scope for this task)

- **Plan Stage 2** — the three-way merge of `pi0_pytorch.py` (patches / fork / runtime)
  and vendoring `model.py` + `array_typing.py`, and switching the package onto its own
  `pi05_infer/gemma`. Not attempted: it changes behaviour (in particular the
  `array_typing` typecheck patch is currently *not* applied and enabling it would move
  e2e by ~3.2 ms), which conflicts with this task's "no behaviour changes /
  reproduce 43.7 ms" gate. Do it as its own change with its own A/B.
- **Plan Stage 3** — rewiring RLinf's `openpi_action_model.py` onto this engine.
  Explicitly excluded by the brief.
- **Plan Stage 4** — cleaning the container's 11 `.bak_*` files, restoring
  site-packages to pristine, adding `pi05-infer` to `install.sh`.
