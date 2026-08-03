# Extraction notes

Every deviation from a straight copy, every coupling that remains, and everything
that could not be cleanly separated. Read this before changing anything in
`pi05_infer/`.

Source of truth for all extractions: `RLinf-pi05-nsys-profile` at rev **`cbb9d2fc`**
(`perf(pi0.5): wire denoise bs=1 optimizations into the inference path`). Verified
byte-identical to the checkout deployed in the `pi05bench` container.

---

## 0. Which box a number came from -- two machines, one path prefix

Read this before comparing any two numbers in this file. The extraction ran across
**two machines** whose working trees share a path prefix, so a path alone does not
identify which copy a measurement was taken on:

| namespace | role |
|---|---|
| analysis box | where the git repo lives; no GPU of the target architecture |
| GPU box, host | a second copy of the tree |
| GPU box, inside the benchmark container | a bind mount of the above |

The two copies are kept in sync by `tar \| ssh`; the git history lives on the analysis
box and is mirrored to the GPU box.

**This is why the notes below keep saying which box a number came from.** It is the
specific reason several early figures could not be reconciled: they were taken on
different machines under the same-looking path.

It is also why the byte-identity claim above rests on md5 rather than on a rev. The
GPU box's copy of `RLinf-pi05-nsys-profile` is **not a git repository** -- it is an
unpacked tree, so `git` cannot be run against it. Its `openpi_action_model.py` is
md5 `86e98eeb580dc7a74b68db3fab1e3866`, identical to the analysis box's `cbb9d2fc`, so
the two are the same code. The analysis box's working tree additionally carries
uncommitted WIP (`denoise_static_masks` + `max_token_len=48`, +49 lines) which is
**not** in the container and therefore **not** part of the measured baseline or of
this extraction.

---

## 1. Stage 0 -- rescue (commit 1)

Four files were copied verbatim (`docker cp`) out of the container's site-packages,
where they were untracked by any repository and one `install.sh` re-run away from
being destroyed:

| repo path | container source | lines | md5 |
|---|---|--:|---|
| `pi05_infer/gemma/modeling_gemma.py` | `transformers/models/gemma/modeling_gemma.py` | 1107 | `5beeb6cb...` |
| `pi05_infer/gemma/rlinf_fused_denoise.py` | `transformers/models/gemma/rlinf_fused_denoise.py` | 552 | `dbbf4c44...` |
| `pi05_infer/openpi_patched/pi0_pytorch.py` | `openpi/models_pytorch/pi0_pytorch.py` | 480 | `91a4a79c...` |
| `pi05_infer/openpi_patched/gemma_pytorch.py` | `openpi/models_pytorch/gemma_pytorch.py` | 285 | `ea915248...` |

At Stage 1 these were rescue copies only: not packages, not imported, and
`pi05_infer` still resolved `transformers.models.gemma` and
`openpi.models_pytorch.pi0_pytorch` from the container's site-packages. **Stage 2
(section 8) wired the package onto them**; the four files above are now the code that
actually runs, and the table above is their provenance.

---

## 2. What was dropped from `openpi_action_model.py` (1824 lines -> `engine.py`)

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

Config: `OpenPi0Config` (34 fields) -> `OpenPi0InferConfig` (8 fields). Retained
fields keep their original names and defaults.

### 2.1 DELIBERATE EXCEPTIONS -- RL code that was NOT dropped

Three statements are RL instrumentation but sit **inside one denoise step**, i.e.
inside the region that the "238 kernels/step" regression gate measures and inside the
body captured by the Stage-1 CUDA graph. Deleting them would change the kernel count
and invalidate the A/B against the container. They are computed and the results
discarded:

1. `get_logprob_norm(x_t, x_t_mean, x_t_std)` in the eager denoise loop, and inside
   `_capture_denoise_step._step`.
2. `value_t = torch.zeros((bsize), device=device)` in `sample_mean_var_val` -- in the
   original this is the `else` branch of the value-head `if`, which is always taken
   when `add_value_head=False` (the inference config). Kept as an unconditional
   statement, producing the identical device tensor.
3. `self._get_noise_level(device, dtype)` in `sample_mean_var_val` -- its return value
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
| `engine.py` `__init__` | `self.logger = get_logger()` -> module `logging.getLogger(__name__)` | `rlinf.utils.logging` is not a dependency of this package |
| `engine.py` | `from rlinf.utils.utils import nvtx_range` -> `pi05_infer._vendored.nvtx` | `rlinf.utils.utils` imports `rlinf.scheduler.Worker` (Ray) at module scope |
| `engine.py` | `PI0Pytorch, BasePolicy` base classes kept; `ForwardType` no longer imported | no dispatcher |
| `builder.py` | `get_model(cfg: DictConfig)` -> `build_model(**kwargs)` with a thin `get_model(cfg)` shim | removes the omegaconf dependency from the core path without breaking the existing call shape |
| `builder.py` | `repack_transforms = transforms.Group()` and `default_prompt = None` inlined | both were dead locals in the original (`*Group().inputs` is empty) |
| `builder.py` | `norm_stats = None; if norm_stats is None:` unwrapped | dead conditional in the original |
| `bench/standalone_infer_bench.py` | `predict_action_batch(env_obs, mode="eval")` -> `predict_action_batch(env_obs)`; result unpacking removed | new return shape |
| `bench/standalone_infer_bench.py` | added `--dump-actions` and `--clocks-json` | verification support; both no-ops unless passed |
| `_vendored/nvtx.py` | the two fallback warnings use `logging` instead of `rlinf.utils.logging.get_logger` | breaking the RLinf import |
| `_vendored/base_policy.py` | one import line rewritten to `pi05_infer._vendored.cuda_graph` | verbatim otherwise |
| `dataconfig/{turtle,libero}_dataconfig.py` | one import line each rewritten to `pi05_infer.dataconfig.policies` | verbatim otherwise |
| `tools/bitgate.py` | the hard-coded gemma directory became `sys.argv[1]` with the original as default | so the rescued copy can be gated too |
| `tools/ksum.py` | none | verbatim from the internal kernel-fusion scripts |

`tools/prof.sh`, `tools/stream_summary.py`, `tools/denoise_kernels.py` and
`tools/ab_rlinf_reference.py` are new (written for this extraction's verification);
`prof.sh` is adapted from `kernel_fusion/scripts/prof.sh`.

`_vendored/cuda_graph.py`, `dataconfig/policies/{turtle,libero}_policy.py` and
`dataconfig/policies/__init__.py` are byte-identical to their RLinf originals.

## 4. `dataconfig` -- minimal subset only

`pi05_infer/dataconfig/__init__.py` keeps 3 of the 16 RLinf `TrainConfig`s
(`pi0_libero`, `pi05_libero`, `pi05_turtle`) and their two data configs / policies.
The other 13 (maniskill, metaworld, calvin, robocasa, robotwin_aloha, franka,
franka_co_training, dual_franka_tcp_rot6d, polaris, isaaclab, gsenv, behavior,
realworld) are **not** vendored. `get_openpi_config`, `_override_with_model_path`
and `_override_with_data_kwargs` are verbatim.

RLinf's own `dataconfig/` is untouched -- it is shared by `openpi_cfg` and three
workers and cannot be moved. This is a deliberate duplicate; the two will drift.
`rlinf/models/embodiment/openpi/transforms/` (which needs `rlinf.utils.rot6d`) is not
reachable from the two retained data configs and was not vendored.

## 5. Couplings that remain

1. **openpi.** `engine.py` imports `openpi.transforms`, `openpi.models.model` and
   `openpi.models.pi0_config.Pi0Config`, and subclasses `PI0Pytorch` -- since Stage 2
   the *vendored* `pi05_infer.openpi_patched.pi0_pytorch.PI0Pytorch`, which itself
   still imports `openpi.models.gemma` and `openpi.models_pytorch.preprocessing_pytorch`.
   This is by design: openpi stays installed, for `Observation`, transforms and
   checkpoint loading.
2. ~~**The patched site-packages.**~~ **Closed by Stage 2** for the expert; see section 8.
   The remaining site-packages dependency is openpi's own `transformers_replace`
   patch, which the **PaliGemma prefix** needs -- section 8.4.
3. **`RLINF_DISABLE_OPENPI_TYPECHECK`.** The env-var gate that no-ops openpi's
   `@at.typecheck` is copied as-is, including its `RLINF_` prefix, so a single env var
   still controls both codebases. Default off in both, so both arms of the A/B pay the
   same cost -- which is what makes it harmless to the A/B, and is the only property
   relied on here. **The size of that cost is unsettled: `engine.py` quotes ~2 ms and
   this file quotes ~3.2 ms.** The two figures come from different sessions and it has
   not been re-measured, so read it as "2-3 ms of pure CPU" until someone re-runs it.
4. **`self.sample_actions = sample_actions_func` in `__init__`.** The rebind that
   stops `PI0Pytorch` from polymorphically calling the subclass method is preserved
   verbatim. Do not "clean it up".
5. **`tools/ab_rlinf_reference.py`** imports RLinf (read-only) -- it exists purely to
   run the reference arm of the A/B and is not part of the package.

## 6. Things measurement could not settle cleanly

- **"238 kernels/step".** With `tools/denoise_kernels.py` both arms measure **234.90**.
  The tool's window runs from the first to the last stream-157 kernel of a predict, so
  the trailing eager glue of the final denoise step falls outside it; extending the
  window past that point picks up the next predict's prefix instead. Since the RLinf
  arm -- the code the 238 figure was recorded against -- also measures 234.90, the gap
  is a definition difference in the counting method, not a regression. If you need the
  exact historical number back, find the script that produced it; it was not among
  the internal kernel-fusion scripts.
- **"1236 us/step" denoise.** Under nsys with `--gpu-metrics-devices` the measured GPU
  busy time is 1135.8 us/step on stream 157 alone and 1613 us/step summing streams 157
  and 158 (which overlap). Neither reproduces 1236 us directly; both are identical
  between the two arms to within 0.07 %, which is the property that matters here.
- **Prefix busy time** is 0.39 % higher in arm B (23863.9 vs 23771.8 us/predict) on a
  *strictly smaller* kernel set with an identical kernel-category table. This is
  run-to-run noise, not a regression, but it was a single paired run -- if you touch the
  prefix, re-measure with more repetitions.
- **Only one paired A/B run** was done per arm (30 timed iterations each), and the
  reference arm ran first, i.e. on a colder card. The 0.52 ms gap is consistent with
  the 6 removed bookkeeping kernels but is not resolved beyond the +/-0.7 ms rebuild
  variance by a single pair.

### 6.1 `36` prefix-KV D2D copies per predict is the *optimised* signature -- not a regression

Investigated 2026-07-28 after a profile of this package was read as evidence
that the static KV buffer had been lost in the extraction. It had not. Recorded here because
the number invites exactly that misreading a second time.

The profile shows 1368 `copyKind=8` copies of 495 616 B over 38 predicts -- **36/predict**,
in one contiguous burst per predict (all 38 bursts are exactly 36 wide). 495 616 B =
968 prefix tokens x 256 head_dim x 2 B, and 36 = 18 layers x {K, V}. That is
`GemmaAttention.prime_kv_static` filling the prefix half of each layer's static buffer
**once per predict**, which is precisely what the optimisation is supposed to cost: the
prefix is rebuilt from a new observation every predict, so this can never be zero. The
scale to compare against is:

| prefix-sized D2D copies / predict | meaning |
|--:|---|
| **360** | `torch.cat` branch -- the prefix re-materialised on all 10 denoise steps (un-optimised) |
| **36** | static KV buffer active -- prefix written once, before the loop |
| 0 | not reachable; would require the prefix not to change between predicts |

Runtime probe (eager build, one predict, both arms): `prime_kv_static` called once,
**18/18 layers primed** at shape `(1, 1, 1018, 256)` with `_kv_prefix_len = 968`, all
**180** attention calls (18 layers x 10 steps) take the static branch and **0** take
`torch.cat`; device `copy_` sizes are `{495616: 36, 25600: 360}` -- the 36 prefix writes
plus the 360 (= 18 x 10 x 2) 50-token suffix tails. The **RLinf reference arm measures the
same thing**, as it must: `pi05_infer/gemma/modeling_gemma.py` differs from the container's
patched `transformers/models/gemma/modeling_gemma.py` only in imports and comments.

In the compiled build the 360 suffix copies disappear into `_qkv_rope_kernel` (6840 in the
profile = 38 predicts x 10 steps x 18 layers), leaving only the 36. Note that a live fused
kernel is *by itself* proof that the buffer is primed: the fused branch is guarded on
`self.kv_static_k is not None`, so it cannot run while the static path is inactive. The two
can never disagree.

The historical **-0.51 ms** for this optimisation stands. It was a plain wall-clock pair
(45.61 -> 45.10 ms), not an nsys number. The separately recorded "KV memcpy 42.6 -> 0/predict"
is a **different metric** that has been conflated with it: copies counted *inside the denoise
loop only*, and credited to the Stage-1 CUDA graph, not to the static buffer.

### 6.2 `_copy_kv_into_static` is dead work on the Stage-1 graph path (unmeasured, not changed)

Code-evident, noticed during the above and left alone. On the graph path
`_refresh_denoise_inputs` -> `_copy_kv_into_static` copies the fresh prefix cache into the
`DynamicCache` that was captured as a graph input -- another 36 x 495 616 B ~ 17.8 MB/predict.
But the captured expert forward reads `self.kv_static_k`, and touches `past_key_value` only
for an `is not None` test, so nothing ever reads what that copy wrote; `prime_kv_static`
-- called from `sample_actions` earlier in the same predict, *before*
`_refresh_denoise_inputs` -- is what actually supplies the fresh prefix. A
prior session measured this transfer at sub-0.1 ms, so the upside is small and deleting it
would have to be re-validated against the graph capture.

**Updated 2026-07-28:** `bench/standalone_infer_bench.py --stage1` now *does* call
`capture_cuda_graph`, so this path is exercised and appears in
the 2026-07-28 Stage-1 profile. The dead-copy observation stands and
is still not acted on; the measured Stage-1 win (-0.93 ms/predict, paired) is large enough
that a sub-0.1 ms follow-up is not worth risking the capture over.

## 7. Not done (out of scope for this extraction)

- **The rest of the vendoring** -- the *three-way merge* of `pi0_pytorch.py`
  (patches / fork / runtime) and vendoring `model.py` + `array_typing.py`. Still not
  attempted: it changes behaviour (in particular the `array_typing` typecheck patch is
  currently *not* applied, and enabling it would move e2e by the 2-3 ms of section 5 item 3),
  which conflicts with the "no behaviour change" gate. Do it as its own change with its
  own A/B. The *first* half -- switching the package onto its own `pi05_infer/gemma`
  and `pi05_infer/openpi_patched` -- is done; see section 8.
- **Rewiring RLinf's `openpi_action_model.py` onto this engine.** Out of scope: this
  repository does not modify RLinf, and the RLinf path is needed unchanged as the
  reference arm of the A/B.
- **Restoring the benchmark container's site-packages to pristine** (cleaning its 11
  `.bak_*` files, adding `pi05-infer` to `install.sh`). Section 8 is the prerequisite
  and is now in place, but the restore itself was deliberately **not** done: the
  container's patched `transformers/models/gemma/` is the reference arm of the A/B.

---

## 8. Stage 2 (first half) -- the package now runs its own model code

Before this change, `pi05_infer/gemma/` and `pi05_infer/openpi_patched/` were inert
rescue copies and the running code came from the container's globally-overwritten
`transformers` / `openpi`. Now `import pi05_infer` pulls in the vendored files.

### 8.1 The edits

Six changes, all imports / module plumbing except the one in 8.3:

| file | change |
|---|---|
| `pi05_infer/gemma/__init__.py` | **new** -- makes the directory a package (needed for `modeling_gemma`'s `from . import rlinf_fused_denoise`) |
| `pi05_infer/gemma/modeling_gemma.py` | the 12 `...`-relative + 1 `.`-relative imports -> absolute `transformers.*` imports. No other line touched. |
| `pi05_infer/gemma/rlinf_fused_denoise.py` | custom-op registration namespace `rlinf::` -> `pi05_infer::` (see 8.3) |
| `pi05_infer/openpi_patched/__init__.py` | **new** -- makes the directory a package |
| `pi05_infer/openpi_patched/gemma_pytorch.py` | `from transformers import GemmaForCausalLM` -> `from pi05_infer.gemma.modeling_gemma import GemmaForCausalLM` |
| `pi05_infer/openpi_patched/pi0_pytorch.py` | `from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel` -> `from pi05_infer.openpi_patched.gemma_pytorch import ...` |
| `pi05_infer/engine.py` | `from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks` -> `from pi05_infer.openpi_patched.pi0_pytorch import ...` |

Not one line of kernel code, model code, tile config or numerics changed. `BLOCK_K`
stays 64 (GeGLU) / 128 (QKV); `RLINF_FUSE_GEGLU`, `RLINF_FUSE_QKV_ROPE`,
`RLINF_FUSE_*_CFG` and `RLINF_FUSE_GEGLU_MAX_M` keep their names and defaults; both
PyTorch fallbacks are untouched.

> **Naming, 2026-07-31.** The gate/up fusion was called *SwiGLU* everywhere until
> this date. That was a misnomer: SwiGLU is `silu(gate) * up`, while Gemma -- and
> this kernel -- computes `gelu_tanh(gate) * up`, i.e. **GeGLU**
> (`hidden_act = "gelu_pytorch_tanh"`). Code, ops and docs have been renamed; no
> number, guard or numeric behaviour changed. The pre-rename environment variables
> (`RLINF_FUSE_SWIGLU`, `RLINF_FUSE_SWIGLU_CFG`, `RLINF_FUSE_SWIGLU_MAX_M`) are
> still accepted as aliases, and history before this date reads `SwiGLU` throughout.

### 8.2 Vendored vs. imported from `transformers` / `openpi`

The rule applied: **import anything we did not modify.** `modeling_gemma.py` is a
whole-file fork, so it necessarily carries unmodified classes too
(`GemmaRotaryEmbedding`, `GemmaForCausalLM`, `GemmaForSequenceClassification`,
`GemmaForTokenClassification`, `eager_attention_forward`, `apply_rotary_pos_emb`,
`_gated_residual`) -- those are not "vendored decisions", they are the file's own
contents. Everything the file *references* is imported:

| symbol | source | why |
|---|---|---|
| `GemmaConfig` | `transformers.models.gemma.configuration_gemma` | not modified by us (it *is* modified by openpi's `transformers_replace`, which adds `use_adarms` / `adarms_cond_dim`) |
| `PreTrainedModel`, `ALL_ATTENTION_FUNCTIONS` | `transformers.modeling_utils` | unmodified |
| `GradientCheckpointingLayer` | `transformers.modeling_layers` | unmodified |
| `GenerationMixin` | `transformers.generation` | unmodified |
| `create_causal_mask` | `transformers.masking_utils` | unmodified |
| `FlashAttentionKwargs` | `transformers.modeling_flash_attention_utils` | unmodified |
| `BaseModelOutputWithPast` &c. | `transformers.modeling_outputs` | unmodified |
| `ROPE_INIT_FUNCTIONS`, `dynamic_rope_update` | `transformers.modeling_rope_utils` | unmodified |
| `ACT2FN` | `transformers.activations` | unmodified |
| `Cache`, `DynamicCache` | `transformers.cache_utils` | unmodified |
| `Unpack` | `transformers.processing_utils` | unmodified |
| `LossKwargs`, `auto_docstring`, `can_return_tuple`, `logging` | `transformers.utils` | unmodified |
| `PaliGemmaForConditionalGeneration`, `CONFIG_MAPPING` | `transformers` | the **prefix**; deliberately stock |
| `modeling_gemma.{apply_rotary_pos_emb, eager_attention_forward, _gated_residual}` used by `gemma_pytorch.py` | `transformers.models.gemma` | unmodified by us, and only reached on the joint prefix+suffix *training* path, which inference never takes |
| `openpi.models.gemma`, `openpi.models_pytorch.preprocessing_pytorch` | openpi | unmodified |
| `openpi.transforms`, `openpi.models.model`, `openpi.models.pi0_config` | openpi | unmodified |

Only two openpi files carry local modifications and hence are vendored:
`pi0_pytorch.py` (batched SigLIP; device-side `att_masks`) and `gemma_pytorch.py`
(`adarms_mod` plumbing; expert construction). Everything else in openpi is imported.

### 8.3 The one non-import change: custom-op namespace

`torch.library` namespaces are **process-global**. `transformers.models.gemma` is
still imported in every run (the prefix needs it), and in the current container that
copy still does `from . import rlinf_fused_denoise`, registering `rlinf::gate_up_geglu`
and `rlinf::qkv_rope_kv`. Registering the same names from `pi05_infer/gemma` would
raise; `modeling_gemma` catches that (`except Exception: _FUSED_OPS = None`), so the
failure mode is **silent loss of both fusions** in whichever copy imports second --
i.e. an unannounced perf regression, exactly the class of bug this extraction exists to
remove. The vendored ops are therefore registered as `pi05_infer::gate_up_geglu` /
`pi05_infer::qkv_rope_kv`. Op *name*, kernel source, tile configs, guards, fallbacks
and env-var names are unchanged; only the registration namespace differs, and the
GPU-side kernel names in nsys are unaffected.

### 8.4 What this does NOT isolate us from (the boundary)

**openpi's `transformers_replace` patch is still required -- by the prefix.** openpi
ships modified `transformers/models/{gemma,paligemma,siglip}` files and its own
`install.sh` copies them into site-packages; `pi0_pytorch.py` even asserts they are
installed. Those files add the adaRMS API (`GemmaRMSNorm.forward` returning
`(hidden, gate)`, `adarms_cond` on the model/layer forwards, `_gated_residual`,
`use_adarms` on `GemmaConfig`). The PaliGemma prefix is built by
`PaliGemmaForConditionalGeneration` -> `AutoModel.from_config`, i.e. from
site-packages, and `gemma_pytorch.py` calls it with `adarms_cond=`. On a *vanilla
upstream* transformers 4.53.2 that call would break.

So "pristine site-packages" here means **openpi-installed** (transformers +
`transformers_replace`), not upstream-vanilla. That is the deliberate boundary:

* **removed** -- the dependency on *our* +245-line overwrite of
  `transformers/models/gemma/modeling_gemma.py` and on the copy of
  `rlinf_fused_denoise.py` next to it. That overwrite was never part of any install
  chain and was the actual hazard.
* **kept** -- the dependency on openpi's own patch, which openpi installs itself and
  which the *prefix* (not the expert) needs.

Isolating the prefix too would mean vendoring PaliGemma + SigLIP + a second Gemma,
i.e. re-vendoring most of openpi's `transformers_replace` for no benefit: the prefix
is the part we did **not** modify, and keeping it on the shared copy is what makes
"expert != prefix" checkable in one line.

Verified on the analysis box, whose transformers has `transformers_replace` applied
but **not** our overwrite (no `_FUSED_OPS` in `transformers.models.gemma`): the
package imports, builds the expert from `pi05_infer.gemma`, and registers both custom
ops. That is the "pristine site-packages" case.

Two consequences of leaving the container as-is, both harmless:

1. Both copies of `rlinf_fused_denoise.py` are imported and both compile their Triton
   kernels -- but only on first use, and the prefix never reaches a fused kernel
   (`_GEGLU_MAX_M = 64` < 968, so `fused_gate_up_geglu` returns `None` and the
   prefix falls back to eager). Identical to the measured baseline.
2. When site-packages is eventually restored, the prefix loses even that per-layer
   `None`-returning check. Strictly less CPU work; nothing else changes.

### 8.5 Verification

Run on the GPU box, RTX PRO 5000 72 GB Blackwell (sm_120), 300 W cap, one job at a
time, `pi05_turtle`, bs=1, 10 denoise steps, 3 x 128^2 cameras, `max-autotune`.

**Isolation** (`tools/isolation_check.py`). It builds the real model the way
`bench/standalone_infer_bench.py` does, prints the defining module of every module
instance in both towers, and asserts them. What it checked and what it found:

| check | required | found |
|---|---|---|
| expert `ForCausalLM`, `GemmaModel`, decoder layer, attention, MLP, RMSNorm | a `pi05_infer.gemma.*` module | `pi05_infer.gemma.modeling_gemma` |
| prefix `GemmaModel`, decoder layer, attention, MLP, RMSNorm | a `transformers.*` module | `transformers.models.gemma.modeling_gemma` |
| prefix PaliGemma | printed, not asserted | `transformers.models.paligemma.modeling_paligemma` |
| prefix vision tower | printed, not asserted | `transformers.models.siglip.modeling_siglip` |
| expert `GemmaModel` exposes `build_adarms_stack`, `build_qkv_fused`, `prime_kv_static`, `clear_kv_static`, `refresh_derived_weights` | all five present | present |
| expert `_FUSED_OPS.__file__` | must sit beside the vendored `modeling_gemma` | `<repo>/pi05_infer/gemma/rlinf_fused_denoise.py` |
| `torch.ops.pi05_infer.gate_up_geglu`, `torch.ops.pi05_infer.qkv_rope_kv` | both registered | both `True` |
| `transformers.models.gemma.modeling_gemma.{__file__, _FUSED_OPS}` | printed, not asserted -- shows whether the container still carries the old global overwrite | -- |

The run ended `ISOLATION_OK`. The row set is deliberately described rather than pasted:
its stdout changes whenever a check is added, and a pasted transcript goes stale silently.

**Bit-exactness**

| check | result |
|---|---|
| `tools/bitgate.py` (now defaults to `pi05_infer/gemma`) -- GeGLU vs inductor | bitwise equal, `max|delta | = 0.00e+00`, 0/204800 elems differing |
| ... QKV+RoPE q / k / v vs inductor | bitwise equal, `max|delta | = 0.00e+00`; q strides identical `(102400,256,2048,1)` |
| end-to-end actions `[1,50,6]` float64, fixed seed, vs the RLinf path | **bitwise equal, `max|delta | = 0.00e+00`** |

**Latency** -- paired, alternating B A B A in one session, 30 timed iterations each
after 8 warmups, plain wall clock, GPU otherwise idle:

| run | arm | mean | p50 | min / max | SM clock, power |
|---|---|--:|--:|--:|---|
| B1 | `pi05_infer` (vendored) | 44.29 | 44.27 | 43.52 / 45.21 | 2445 MHz (2422-2460), 240.1 W |
| A1 | RLinf reference | 44.37 | 44.21 | 43.75 / 45.32 | not sampled |
| B2 | `pi05_infer` (vendored) | 44.05 | 44.03 | 43.56 / 44.42 | 2438 MHz (2370-2452), 208.8 W |
| A2 | RLinf reference | 44.97 | 44.91 | 44.34 / 45.70 | not sampled |
| -- | **B mean 44.17 vs A mean 44.67** | | | | delta = -0.50 ms |

The pre-change measurement was B 44.01 / A 44.53 at 2445 MHz, i.e. delta = -0.52 ms. The
paired gap is reproduced to 0.02 ms; the absolute B number moved +0.16 ms, well inside
the documented +/-0.7 ms rebuild variance, at a comparable clock.

**Kernels** -- nsys 2026.1.2, `-t cuda,nvtx --cuda-graph-trace=node
--gpu-metrics-devices=cuda-visible`, 12 predicts inside `cudaProfilerApi`, exported to
sqlite with the same binary. `_s2` = after this change; the pre-change profiles from
the previous task were kept and re-counted with the same tools.

| | `pi05_infer` pre | `pi05_infer` post | RLinf pre | RLinf post |
|---|--:|--:|--:|--:|
| denoise, kernels/step (`denoise_kernels.py`) | 234.90 | **234.90** | 234.90 | 234.90 |
| ... of which stream 157 | 171.00 | **171.00** | 171.00 | 171.00 |
| **prefix, stream 7, kernels/predict** | 1018.00 | **1018.00** | 1024.00 | 1024.00 |
| stream 158, kernels/step | 35.60 | 35.60 | 35.60 | 35.60 |
| total kernels/predict | 3084 | **3084** | 3090 | 3090 |
| stream 7 GPU busy, us/predict | 23863.9 | 23914.9 (+0.21 %) | 23771.8 | 24033.6 (+1.10 %) |
| stream 157 GPU busy, us/step | 11366.1 | 11398.6 (+0.29 %) | 11358.4 | 11357.2 |

The per-kernel-category count tables (`tools/ksum.py`, category x count x distinct
names) are **byte-identical pre vs post** on streams 7, 157 and 158 for *both* arms.
The busy-time deltas are run-to-run noise -- note the reference arm, which did not
change at all, moved 5x more than ours.

Stream 157 still shows `_geglu_mm_kernel` 180.00/step and `_qkv_rope_kernel`
180.00/step, i.e. both fusions are live on the expert through the vendored module.
Inductor names the wrapper kernel after the op *name*, not the namespace, so
`triton_per_fused__to_copy_add_gate_up_geglu_mean_mul_pow_rsqrt` tracks the op name
(it read `..._gate_up_swiglu_...` before the 2026-07-31 rename) while the counts remain
directly comparable with the pre-change profiles.

The only kernel-count differences anywhere are the 6 per-predict RL-bookkeeping
kernels between `pi05_infer` and RLinf that section 6 already documented (`reduce_kernel` -1,
`index_elementwise_kernel` -1, -4 across the `torch.stack` copies) -- identical before
and after this change.

The caveat in section 6 about "238 kernels/step" is unaffected: all four profiles measure
234.90 with `tools/denoise_kernels.py`, including the two RLinf arms, so the gap
remains a counting-definition difference rather than a regression.

