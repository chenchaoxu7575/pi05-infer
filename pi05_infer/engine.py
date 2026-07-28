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
"""Pure pi0.5 / pi0 inference engine, extracted from RLinf.

Source: ``rlinf/models/embodiment/openpi/openpi_action_model.py`` at rev
``cbb9d2fc`` (1824 lines; verbatim copy kept in ``_extract_src/`` for diffing).

What was kept: the inference path only --
``predict_action_batch`` -> ``sample_actions`` -> ``sample_mean_var_val`` ->
``get_velocity`` -> ``get_suffix_out``, plus ``_build_prefix_cache``,
``enable_torch_compile``, the Stage-1 denoise CUDA-graph capture/replay, the
adaRMS modulation table, and ``invalidate_weight_derived_caches``.

What was dropped: everything RL. See ``EXTRACTION_NOTES.md`` for the full list
and for the two hot-path lines that were deliberately NOT dropped.

**Hot-path fidelity rule.** Every statement inside one denoise step is byte-identical
to the RLinf original, including ``get_logprob_norm`` and the zero-valued ``value_t``.
Those two are RL instrumentation, but they sit inside the measured per-step region (and
inside the captured CUDA graph), so deleting them would change the kernel count that the
238-kernels/step regression gate measures. They are computed and discarded.
Deletions are confined to code outside the per-step body.
"""

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

# Opt-in: no-op openpi's ``@at.typecheck`` (jaxtyped + beartype runtime
# type/shape validation on the ``Observation`` dataclass). On the eager PyTorch
# rollout path it costs ~2ms/predict of pure CPU with the GPU idle (verified
# bit-exact: the check never changes values). Rollout obs shapes are stable, so
# this is safe once the pipeline is validated. Must run BEFORE openpi model
# classes are decorated at import, hence before the openpi imports below.
# Default off (typecheck stays on); set RLINF_DISABLE_OPENPI_TYPECHECK=1 to enable.
if os.environ.get("RLINF_DISABLE_OPENPI_TYPECHECK") == "1":
    import openpi.shared.array_typing as _at

    _at.typecheck = lambda t: t

from openpi import transforms as _transforms  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402

from pi05_infer._vendored.base_policy import BasePolicy  # noqa: E402
from pi05_infer._vendored.nvtx import nvtx_range  # noqa: E402

# The vendored fork, NOT ``openpi.models_pytorch.pi0_pytorch``: this is what makes
# the action expert resolve to ``pi05_infer.gemma`` rather than to whatever
# ``transformers.models.gemma`` the container happens to have. See
# ``pi05_infer/openpi_patched/__init__.py``.
from pi05_infer.openpi_patched.pi0_pytorch import (  # noqa: E402
    PI0Pytorch,
    make_att_2d_masks,
)

logger = logging.getLogger(__name__)

# Kill switch for the dead-``adarms_cond`` elision in ``get_suffix_out`` (see there).
# Default on; set ``RLINF_SKIP_DEAD_ADARMS_COND=0`` to restore the old behaviour, i.e.
# compute the timestep conditioning on every denoise step even though nothing reads it.
# Read once at import so the branch is a Python constant during CUDA-graph capture.
_SKIP_DEAD_ADARMS_COND = os.environ.get("RLINF_SKIP_DEAD_ADARMS_COND", "1") != "0"


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


@dataclass(frozen=True)
class OpenPi0InferConfig(Pi0Config):
    """Inference-only subset of RLinf's ``OpenPi0Config``.

    The RL-training fields (noise annealing schedule, joint/safe logprob, value-head
    geometry, critic detaching, DSRL and NFT parameter groups) are not present. The
    remaining fields are unchanged, including their defaults.
    """

    # pi0_libero, pi05_libero, pi05_turtle
    config_name: str = "pi0_libero"
    num_images_in_input: int = 2  # number of images in input
    # Retained because ``_get_noise_level`` reads it on the hot path; inference always
    # takes the flow_ode branch regardless of its value.
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    train_expert_only: bool = False


class OpenPi0Inference(PI0Pytorch, BasePolicy):
    """pi0 / pi0.5 action prediction, inference only."""

    config: OpenPi0InferConfig

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            return [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        return [
            "GemmaMLP",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(self, config: OpenPi0InferConfig):
        # Override `sample_actions` to prevent parent class polymorphic call.
        # (Keep this: PI0Pytorch calls self.sample_actions internally.)
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.logger = logger

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path
            # (e.g. "model.action_in_proj" -> "action_in_proj").
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self.torch_compile_enabled = False
        self._torch_compile_mode = None

        # ===== Denoise-step CUDA graph (Stage 1, inference only) =====
        # Cache for the suffix attention-mask tensor. The parent embed_suffix builds it
        # via torch.tensor(<python list>), a host->device op that is forbidden while a
        # CUDA graph is capturing. The pattern is fixed (depends only on action_horizon /
        # pi05), so we cache the device tensor on the first call and reuse it thereafter.
        self._suffix_att_masks_cache = None
        # Lazily-captured single flow_ode denoise step. Populated on the first eval-shaped
        # sample_actions call when cuda_graph_manager is set (see capture_cuda_graph).
        self._denoise_graph_captured = False
        self._denoise_graph_spec = None  # config the graph was captured for
        self._denoise_static = None  # static input buffers + persistent KV cache

    # ------------------------------------------------------------------
    # observation plumbing
    # ------------------------------------------------------------------
    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = tree_map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split
        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0) if len(x.shape) == 3 and transpose else x
                    ),
                    sample,
                )
            else:
                sample = tree_map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)
        # transform
        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
            transformed_samples = list(ex.map(self._input_transform, batch_samples))
        # recombine
        inputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def obs_processor(self, env_obs):
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs["extra_view_images"] is not None:
            processed_obs["observation/extra_view_image"] = env_obs["extra_view_images"]
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    # ------------------------------------------------------------------
    # inference entry point
    # ------------------------------------------------------------------
    def predict_action_batch(self, env_obs, **kwargs) -> torch.Tensor:
        """Env observations -> env-space actions ``[B, action_chunk, action_dim]``."""
        with nvtx_range("predict/obs_processor", color="cyan"):
            to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        with nvtx_range("predict/input_transform", color="cyan"):
            processed_obs = self.input_transform(
                to_process_obs, transpose=False
            )  # policy input obs -> model input obs
        with nvtx_range("predict/precision_processor", color="cyan"):
            processed_obs = self.precision_processor(
                processed_obs
            )  # obs precision processor
        with nvtx_range("predict/observation_from_dict", color="cyan"):
            observation = _model.Observation.from_dict(processed_obs)

        with nvtx_range("predict/sample_actions", color="purple"):
            outputs = self.sample_actions(observation)
        with nvtx_range("predict/output_transform", color="purple"):
            actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"]
        return actions

    @torch.no_grad()
    def sample_actions(self, observation: _model.Observation, noise=None) -> dict:
        """Full inference forward: prefix cache + ``num_steps`` flow_ode denoise steps."""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(self.action_in_proj.weight.dtype)

        with nvtx_range("denoise/preprocess", color="cyan"):
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )

        with nvtx_range("denoise/prefix_cache", color="green"):
            _prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
                images, img_masks, lang_tokens, lang_masks
            )

        x_t = noise

        # Build the adaRMS modulation table BEFORE any CUDA-graph capture: it calls eager
        # embed_suffix/dense (capture-illegal). Cached, so it runs once; the captured denoise
        # step then only gathers from this static table (device gather, capture-safe).
        self._get_adarms_table(state, noise, num_steps, device)

        # Prime the static KV buffers once per predict, so each denoise step writes only the
        # 50 new suffix tokens instead of re-concatenating the whole 968-token prefix.
        expert = self.paligemma_with_expert.gemma_expert.model
        if hasattr(expert, "prime_kv_static"):
            expert.prime_kv_static(past_key_values, self.config.action_horizon)

        # Denoise-step CUDA graph (Stage 1): replay a captured single flow_ode step for
        # every step. Lossless.
        use_denoise_graph = self.is_cuda_graph_enabled() and self._ensure_denoise_graph(
            x_t, state, prefix_pad_masks, past_key_values, num_steps
        )
        if use_denoise_graph:
            self._refresh_denoise_inputs(state, prefix_pad_masks, past_key_values)

        # denoise step
        with nvtx_range("denoise/loop", color="orange"):
            for idx in range(num_steps):
                with nvtx_range("denoise/step", color="orange"):
                    if use_denoise_graph:
                        # Wide-graph replay: expert + value + Euler + logprob in one launch.
                        # Draw sample_noise eagerly first so RNG consumption matches the eager
                        # path (its result is unused since flow_ode std == 0).
                        self.sample_noise(x_t.shape, device)
                        x_t, _log_prob, _value_t = self._replay_denoise_step(x_t, idx)
                    else:
                        x_t_mean, x_t_std, _value_t, _v_t = self.sample_mean_var_val(
                            x_t, idx, state, prefix_pad_masks, past_key_values, num_steps
                        )
                        # Euler step - new tensor assignment instead of an in-place op.
                        x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
                        # RL instrumentation retained for hot-path parity (see module docstring).
                        self.get_logprob_norm(x_t, x_t_mean, x_t_std)
        return {"actions": x_t}

    # ------------------------------------------------------------------
    # denoise internals
    # ------------------------------------------------------------------
    def _get_timesteps(self, denoise_steps, device):
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.zeros((1), device=device)])
        return timesteps

    def invalidate_weight_derived_caches(self):
        """Drop every cache derived from the model weights.

        **Call this after an RL rollout weight sync.** In-place weight updates keep the captured
        CUDA graph valid (it references addresses, not values), but these derived tensors are
        computed FROM the weights and would otherwise silently keep serving stale values:
          - ``_adarms_table``: the precomputed per-timestep adaRMS modulations (built from the
            37 ``dense`` weights),
          - the expert's stacked adaRMS weight and fused QKV weights.
        Cheap: the table is rebuilt lazily on the next ``sample_actions`` (one-off, ~ms).
        """
        expert = self.paligemma_with_expert.gemma_expert.model
        if hasattr(expert, "refresh_derived_weights"):
            # In-place refresh: keeps tensor addresses stable so an already-captured CUDA graph
            # (which references these buffers) stays valid and simply reads the new values.
            expert.refresh_derived_weights()
        # Same rule for the modulation table: recompute from the NEW weights and refill in place.
        table = getattr(self, "_adarms_table", None)
        refs = getattr(self, "_adarms_refs", None)
        if table is not None and refs is not None:
            table.copy_(
                self._compute_adarms_table(
                    refs[0], refs[1], self._adarms_table_key, table.device
                )
            )

    def _get_adarms_table(self, ref_state, ref_noise, num_steps, device):
        """Precompute (once, cached by num_steps) the adaRMS modulation for every denoise step.

        The 37 per-norm dense(cond) projections depend ONLY on the diffusion timestep (a fixed
        linspace schedule, input-independent), so the whole [num_steps, B, n_norm, 3072] table is
        built once and indexed per step -- removing the 37 memory-bound projections (~3.9ms/predict,
        serial on the critical path) from every denoise step. Built with the real per-norm dense
        modules over the exact timesteps the loop uses (timesteps[s], via the same embed_suffix),
        so table[s, :, i] == dense_i(cond_s) bit-for-bit vs the per-dense reference.
        """
        if (
            getattr(self, "_adarms_table_key", None) == num_steps
            and getattr(self, "_adarms_table", None) is not None
        ):
            return self._adarms_table
        table = self._compute_adarms_table(ref_state, ref_noise, num_steps, device)
        self._adarms_table = table
        self._adarms_table_key = num_steps
        # Keep the reference tensors so the table can be recomputed after a weight sync
        # (see invalidate_weight_derived_caches).
        self._adarms_refs = (ref_state, ref_noise)
        return table

    def _compute_adarms_table(self, ref_state, ref_noise, num_steps, device):
        """Compute the [num_steps, B, n_norm, 3072] modulation table from the CURRENT weights."""
        exp = self.paligemma_with_expert.gemma_expert.model
        norms = []
        for layer in exp.layers:
            norms.append(layer.input_layernorm)
            norms.append(layer.post_attention_layernorm)
        norms.append(exp.norm)
        norms = [n for n in norms if getattr(n, "dense", None) is not None]
        if not norms:
            return None
        bsize = ref_state.shape[0]
        timesteps = self._get_timesteps(num_steps, device)
        rows = []
        for s in range(num_steps):
            _, _, _, cond = self.embed_suffix(
                ref_state, ref_noise, timesteps[s].expand(bsize)
            )
            rows.append(
                torch.stack([n.dense(cond) for n in norms], dim=1)
            )  # [B, n_norm, 3072]
        return torch.stack(rows, dim=0).contiguous()  # [num_steps, B, n_norm, 3072]

    def embed_suffix(self, state, noisy_actions, timestep, skip_adarms_cond: bool = False):
        """CUDA-graph-safe wrapper around the parent ``embed_suffix``.

        The parent builds the suffix attention-mask via ``torch.tensor(<python list>)``
        (``pi0_pytorch.py``), the only host->device tensor construction in this code path
        and one that is forbidden while a CUDA graph is capturing. The mask pattern is fixed
        (depends only on ``action_horizon`` / ``pi05``), so we intercept that single call: on
        the first invocation we build the device tensor normally and cache it; afterwards
        (including during graph capture/replay) we return the cached tensor. The expand to
        batch size that follows on the parent side is a view op and is capture-safe.

        Numerically identical to the parent: the cached tensor holds the exact same values.

        ``skip_adarms_cond`` is forwarded verbatim (see the parent): it drops the dead
        timestep-conditioning computation when the caller already holds ``adarms_mod``.
        """
        orig_tensor = torch.tensor

        def _tensor_shim(data, *args, **kwargs):
            # The att_masks construction is the only list->tensor call reachable here.
            if isinstance(data, list):
                if self._suffix_att_masks_cache is None:
                    self._suffix_att_masks_cache = orig_tensor(data, *args, **kwargs)
                return self._suffix_att_masks_cache
            return orig_tensor(data, *args, **kwargs)

        torch.tensor = _tensor_shim
        try:
            return super().embed_suffix(
                state, noisy_actions, timestep, skip_adarms_cond=skip_adarms_cond
            )
        finally:
            torch.tensor = orig_tensor

    def sample_mean_var_val(
        self, x_t, idx, state, prefix_pad_masks, past_key_values, denoise_steps
    ):
        """One flow_ode denoise step: mean, (zero) std, (zero) value and velocity.

        ``idx`` is an int on the eager path and a device tensor inside the captured graph.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        # adaRMS cache: index the precomputed modulation table for this step, replacing the 37
        # memory-bound dense(cond) projections. Works for BOTH integer idx (eager eval) and tensor
        # idx (Stage-1 captured graph) via a device gather (capture-safe, no host sync).
        # The table is per-timestep so its batch dim is redundant (all rows identical) -> take
        # [:, 0] and gather by idx to get [bsize, n_norm, 3072].
        adarms_mod = None
        table = self._get_adarms_table(state, x_t, denoise_steps, device)
        if isinstance(idx, int):
            idx = torch.full((), idx, device=device).expand(bsize)
        if table is not None:
            adarms_mod = table[:, 0][idx]
        # build parameters
        # Retained for hot-path parity: one tiny device-tensor construction per step, present
        # in the measured baseline. flow_ode does not use its value.
        self._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = self._get_timesteps(denoise_steps, device)
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        v_t, _suffix_out = self.get_velocity(
            state, x_t, t_input, prefix_pad_masks, past_key_values, adarms_mod
        )
        # value prediction (no value head on the inference path)
        value_t = torch.zeros((bsize), device=device)
        # sample mean and variance
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        # flow_ode (the only sampler on the inference path)
        x0_weight = 1 - (t_input - delta)
        x1_weight = t_input - delta
        x_t_std = torch.zeros_like(t_input)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, v_t

    def _get_noise_level(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((), self.config.noise_level, device=device, dtype=dtype)

    def get_suffix_out(
        self, state, prefix_pad_masks, past_key_values, x_t, timestep, adarms_mod=None
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        with nvtx_range("denoise/embed_suffix", color="yellow"):
            # When ``adarms_mod`` is in hand (the precomputed modulation table -- the normal
            # denoise path) the expert never consults ``adarms_cond``: modeling_gemma takes the
            # ``adarms_mod`` branch and the ``dense(cond)`` fallback below it never runs. Tell
            # embed_suffix to skip building it -- that removes the sinusoid (cos/sin) plus the
            # two 1024x1024 time-MLP GEMMs and their two silus from every denoise step.
            # Python-level, so the captured graph traces exactly one of the two shapes.
            skip_cond = _SKIP_DEAD_ADARMS_COND and adarms_mod is not None
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(state, x_t, timestep, skip_adarms_cond=skip_cond)
            )

        with nvtx_range("denoise/suffix_mask_prep", color="white"):
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]

            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )

            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Prepare attention masks
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        with nvtx_range("denoise/expert_forward", color="red"):
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                adarms_mod=[None, adarms_mod],
            )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def get_velocity(
        self, state, x_t, timestep, prefix_pad_masks, past_key_values, adarms_mod=None
    ):
        """Compute velocity prediction v_t and raw suffix_out at a given timestep."""
        suffix_out = self.get_suffix_out(
            state, prefix_pad_masks, past_key_values, x_t, timestep, adarms_mod
        )
        with nvtx_range("denoise/action_out_proj", color="yellow"):
            v_t = self.action_out_proj(suffix_out)
        return v_t, suffix_out

    def _build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        """Embed prefix tokens and compute KV cache for efficient suffix generation."""
        with nvtx_range("prefix/embed_prefix", color="green"):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
        with nvtx_range("prefix/mask_prep", color="white"):
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
                prefix_att_2d_masks
            )
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        with nvtx_range("prefix/vlm_forward", color="blue"):
            (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return prefix_output, prefix_pad_masks, past_key_values

    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5*log(2*pi) - 0.5*((x-mu)/sigma)**2
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def freeze_vlm(self):
        if self.config.train_expert_only:
            # Base freeze: paligemma (SigLIP vision encoder + Gemma)
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

    # ------------------------------------------------------------------
    # BasePolicy hooks that inference does not implement
    # ------------------------------------------------------------------
    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "pi05_infer is an inference-only package; training forwards live in RLinf."
        )

    # ===================================================================
    # Denoise-step CUDA graph (Stage 1): hand-captured single flow_ode step.
    # Inference only, lossless. Targets the launch-bound gemma_expert forward
    # inside the denoise loop (GPU idle waiting for CPU to dispatch ~880 tiny
    # kernels/step). We capture ONE flow_ode step and replay it for every
    # step; the Euler update / logprob stay eager, so RNG consumption and
    # numerics are bit-identical to the eager path.
    # ===================================================================
    def capture_cuda_graph(self, train_batch_size: int, eval_batch_size: int):
        """Wire up denoise-step CUDA-graph capture.

        The actual ``torch.cuda.CUDAGraph`` is captured lazily on the first eval-shaped
        ``sample_actions`` call, when a real prefix KV cache (and thus the exact shapes)
        is available. Here we only create the manager and reset capture state.
        """
        from pi05_infer._vendored.cuda_graph import CUDAGraphManager

        # torch.compile is allowed ALONGSIDE the hand-captured graph as long as it does NOT
        # itself emit an inductor CUDA graph: nesting our graph around compiled-cudagraph code
        # is unsupported. The no-cudagraphs / fusion-only modes are the intended combination --
        # compile fuses kernels (incl. the prefill), our graph eliminates launch overhead in
        # the denoise loop. Reject only the cudagraph-emitting modes.
        cudagraph_modes = {"max-autotune", "reduce-overhead"}
        if self.torch_compile_enabled and self._torch_compile_mode in cudagraph_modes:
            raise RuntimeError(
                "enable_cuda_graph (denoise-step capture) cannot be combined with "
                f"torch_compile_mode='{self._torch_compile_mode}' (emits an inductor CUDA "
                "graph). Use a '*-no-cudagraphs' mode, or disable one of them."
            )
        device = next(self.parameters()).device
        self.cuda_graph_manager = CUDAGraphManager(device=device)
        self._denoise_graph_captured = False
        self._denoise_graph_spec = None
        self._denoise_static = None
        self._denoise_eval_batch_size = eval_batch_size
        self.logger.info(
            "[denoise-cudagraph] manager ready; graph captured lazily on first inference."
        )

    @staticmethod
    def _denoise_kv_pairs(past_key_values):
        """Return a list of (key, value) tensor pairs from either a transformers
        ``DynamicCache`` (``.key_cache`` / ``.value_cache``) or a ``list[(K, V)]``."""
        if hasattr(past_key_values, "key_cache") and hasattr(
            past_key_values, "value_cache"
        ):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))
        return [(kv[0], kv[1]) for kv in past_key_values]

    def _copy_kv_into_static(self, src_cache):
        """Copy a freshly-built prefix KV cache into the persistent static KV buffers
        so the captured graph (which references fixed addresses) sees the new prefix."""
        dst_pairs = self._denoise_kv_pairs(self._denoise_static["past_key_values"])
        src_pairs = self._denoise_kv_pairs(src_cache)
        assert len(dst_pairs) == len(src_pairs), (
            f"KV layer count mismatch: static={len(dst_pairs)} new={len(src_pairs)}"
        )
        for (dk, dv), (sk, sv) in zip(dst_pairs, src_pairs):
            dk.copy_(sk)
            dv.copy_(sv)

    def _denoise_graph_signature(self, x_t, state, prefix_pad_masks, num_steps):
        """Identity of the captured graph: shapes/dtype/flags it is valid for."""
        return (
            tuple(x_t.shape),
            x_t.dtype,
            tuple(state.shape),
            tuple(prefix_pad_masks.shape),
            int(num_steps),
        )

    def _ensure_denoise_graph(
        self, x_t, state, prefix_pad_masks, past_key_values, num_steps
    ):
        """Return True if the denoise-step graph can be replayed for this call,
        capturing it lazily on first use. Falls back to eager (False) if the shapes
        or flags differ from what was captured."""
        sig = self._denoise_graph_signature(x_t, state, prefix_pad_masks, num_steps)
        if self._denoise_graph_captured:
            return self._denoise_graph_spec == sig
        self._capture_denoise_step(
            x_t, state, prefix_pad_masks, past_key_values, num_steps, sig
        )
        return self._denoise_graph_captured

    def _capture_denoise_step(
        self, x_t, state, prefix_pad_masks, past_key_values, num_steps, sig
    ):
        """Capture a single flow_ode ``sample_mean_var_val`` (incl. the full gemma_expert
        forward) into a CUDA graph, using this call's real tensors as the static buffers."""
        from pi05_infer._vendored.cuda_graph import GraphCaptureSpec

        device = x_t.device
        bsize = x_t.shape[0]

        # Static input buffers. state / prefix_pad_masks / KV are per-inference (refreshed
        # via copy_ each call); x_t / idx are per-step (copied each replay by the manager).
        x_t_static = x_t.detach().clone()
        idx_static = torch.zeros(bsize, dtype=torch.long, device=device)
        state_static = state.detach().clone()
        ppm_static = prefix_pad_masks.detach().clone()
        # Keep this call's DynamicCache as the persistent static KV buffer (addresses fixed).
        kv_static = past_key_values

        self._denoise_static = {
            "x_t": x_t_static,
            "idx": idx_static,
            "state": state_static,
            "prefix_pad_masks": ppm_static,
            "past_key_values": kv_static,
            # Pre-built per-step idx buffers (long [bsize]); copied into idx_static on replay.
            "idx_buffers": [
                torch.full((bsize,), i, dtype=torch.long, device=device)
                for i in range(num_steps)
            ],
        }

        def _step(inp):
            # WIDE capture: the full flow_ode step in one graph -- expert forward + value
            # + Euler + logprob -- so a single replay replaces #968's expert-only inductor
            # cudagraph PLUS all the eager glue (Euler/logprob/python) that sits between
            # launches and accounts for ~47% idle inside the denoise loop.
            x_t_mean, x_t_std, value_t, _v_t = self.sample_mean_var_val(
                inp["x_t"],
                inp["idx"],
                inp["state"],
                inp["prefix_pad_masks"],
                inp["past_key_values"],
                num_steps,
            )
            # flow_ode: x_t_std == 0, so the Euler update x_t + noise*std reduces to x_t_mean
            # exactly (algebraically exact). The eager sample_noise draw is kept OUTSIDE the
            # graph so global RNG consumption stays identical to the eager path.
            x_t_next = x_t_mean
            log_prob = self.get_logprob_norm(x_t_next, x_t_mean, x_t_std)
            return {
                "x_t_next": x_t_next,
                "log_prob": log_prob,
                "value_t": value_t,
            }

        capture_spec = GraphCaptureSpec(
            name="denoise_step",
            func=_step,
            inputs={
                "x_t": x_t_static,
                "idx": idx_static,
                "state": state_static,
                "prefix_pad_masks": ppm_static,
                "past_key_values": kv_static,
            },
            external_inputs={"x_t", "idx"},
            warmup_iters=5,
        )
        self.cuda_graph_manager.capture(capture_spec)
        self._denoise_graph_captured = True
        self._denoise_graph_spec = sig
        self.logger.info(
            f"[denoise-cudagraph] captured single flow_ode step for signature={sig}"
        )

    def _refresh_denoise_inputs(self, state, prefix_pad_masks, past_key_values):
        """Copy this inference's per-inference inputs into the static graph buffers."""
        self._denoise_static["state"].copy_(state)
        self._denoise_static["prefix_pad_masks"].copy_(prefix_pad_masks)
        self._copy_kv_into_static(past_key_values)

    def _replay_denoise_step(self, x_t, step_idx):
        """Replay the captured full flow_ode step for ``x_t`` at ``step_idx``; returns cloned
        (x_t_next, log_prob, value_t) -- the graph output buffers are overwritten next replay."""
        out = self.cuda_graph_manager.replay(
            "denoise_step",
            {"x_t": x_t, "idx": self._denoise_static["idx_buffers"][step_idx]},
        )
        return (
            out["x_t_next"].clone(),
            out["log_prob"].clone(),
            out["value_t"].clone(),
        )

    def enable_torch_compile(self, mode: str = "max-autotune"):
        if self.torch_compile_enabled:
            return

        self.paligemma_with_expert.paligemma.model.vision_tower.forward = torch.compile(
            self.paligemma_with_expert.paligemma.model.vision_tower.forward, mode=mode
        )

        # NOTE: paligemma.model.language_model and gemma_expert.model share the same LLM backbone.
        # Enabling cuda graph on both simultaneously causes mysterious crashes (likely due to
        # tensor aliasing in the shared computation graph). We disable cuda graph for
        # paligemma.model.language_model since it is not CPU-bound, while gemma_expert.model
        # benefits more from cuda graph.
        self.paligemma_with_expert.paligemma.model.language_model.forward = (
            torch.compile(
                self.paligemma_with_expert.paligemma.model.language_model.forward,
                mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else mode,
            )
        )
        # adaRMS batched projection: build the stacked dense weight ONCE before compile so
        # dynamo dead-code-eliminates the lazy-build branch in GemmaModel.forward (no graph break).
        self.paligemma_with_expert.gemma_expert.model.build_adarms_stack()
        # Fused QKV: one [q+k+v, hidden] GEMM instead of three skinny ones (k/v are 50x256 at
        # bs=1 -> grid=8, ~8x above their memory floor). Built before compile so the traced
        # graph takes the fused branch.
        self.paligemma_with_expert.gemma_expert.model.build_qkv_fused()
        self.paligemma_with_expert.gemma_expert.model.forward = torch.compile(
            self.paligemma_with_expert.gemma_expert.model.forward,
            mode=mode,
            fullgraph=True,
        )
        self.get_logprob_norm = torch.compile(
            self.get_logprob_norm,
            mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else mode,
        )

        self.torch_compile_enabled = True
        self._torch_compile_mode = mode


# Convenience aliases.
Engine = OpenPi0Inference
EngineConfig = OpenPi0InferConfig

__all__ = [
    "Engine",
    "EngineConfig",
    "OpenPi0Inference",
    "OpenPi0InferConfig",
]
