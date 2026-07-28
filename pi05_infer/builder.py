# Copyright 2025 The RLinf Authors.
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
"""Model construction, extracted from ``rlinf/models/embodiment/openpi/__init__.py``
(``get_model``) at rev ``cbb9d2fc``.

Deviations from the original, all inference-only simplifications:
  * builds ``OpenPi0Inference`` / ``OpenPi0InferConfig`` instead of the RL classes;
  * the DictConfig plumbing is replaced by explicit keyword arguments (``build_model``);
    ``get_model(cfg)`` is kept as a thin omegaconf-shaped shim so the RLinf call site
    and the existing benchmark keep working unchanged;
  * ``repack_transforms`` / ``default_prompt`` were dead locals in the original
    (empty Group / None) and are inlined.
Weight loading, dtype casting and the transform wiring are unchanged.
"""

import glob
import os
from typing import Any, Optional

import torch

from pi05_infer.engine import OpenPi0InferConfig, OpenPi0Inference


def build_model(
    model_path: str,
    config_name: str,
    *,
    data_kwargs: Optional[dict] = None,
    **config_overrides: Any,
) -> OpenPi0Inference:
    """Construct a ready-to-run inference engine.

    Args:
        model_path: Checkpoint dir (safetensors + ``<asset_id>/norm_stats.json``),
            or an FSDP checkpoint dir containing ``model_state_dict/full_weights.pt``.
        config_name: openpi TrainConfig name, e.g. ``pi05_turtle``.
        data_kwargs: Optional overrides applied to the data config.
        **config_overrides: Fields written onto the model config, e.g.
            ``num_steps=10``, ``action_chunk=50``, ``train_expert_only=True``.

    Returns:
        An ``OpenPi0Inference`` on CPU with transforms and norm stats wired up.
        Call ``.to(device).eval()`` and optionally ``.enable_torch_compile()``.
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from pi05_infer.dataconfig import get_openpi_config

    actor_train_config = get_openpi_config(
        config_name, model_path=model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0InferConfig(**actor_model_config.__dict__)
    for key, val in config_overrides.items():
        # ``OpenPi0InferConfig`` is a frozen dataclass; the original writes through
        # ``__dict__`` for the same reason.
        actor_model_config.__dict__[key] = val
    actor_model_config.__dict__["config_name"] = config_name

    # load model
    checkpoint_dir = download.maybe_download(str(model_path))

    # Check if this is a checkpoint directory (saved by FSDP): look for
    # model_state_dict/full_weights.pt (direct) or actor/model_state_dict/full_weights.pt
    # (from the runner).
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model = OpenPi0Inference(actor_model_config)
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        model.load_state_dict(all_state_dict, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    # Load the norm stats from the checkpoint (not the config assets dir) so the policy
    # uses the same normalization stats as the original training process.
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
        ],
    )

    return model


def get_model(cfg, torch_dtype=None) -> OpenPi0Inference:
    """omegaconf-shaped shim with the same signature as RLinf's ``get_model``."""
    openpi_cfg = dict(cfg.openpi) if cfg.openpi is not None else {}
    config_name = openpi_cfg.pop("config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    return build_model(
        model_path=cfg.model_path,
        config_name=config_name,
        data_kwargs=data_kwargs,
        **openpi_cfg,
    )
