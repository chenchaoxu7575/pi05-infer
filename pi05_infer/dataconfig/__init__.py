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
"""Minimal subset of ``rlinf/models/embodiment/openpi/dataconfig`` (rev cbb9d2fc).

Only the three TrainConfigs the pi0.5 bs=1 inference benchmark needs are kept:
``pi0_libero``, ``pi05_libero``, ``pi05_turtle``. The remaining 13 configs
(maniskill, metaworld, calvin, robocasa, robotwin, franka, polaris, isaaclab,
gsenv, behavior, realworld, dual_franka, franka_co_training) and their data
configs / policies are NOT vendored -- see EXTRACTION_NOTES.md.

The three retained ``TrainConfig`` literals and ``get_openpi_config`` /
``_override_with_model_path`` / ``_override_with_data_kwargs`` are copied
verbatim from the RLinf original.
"""

import dataclasses
import difflib
from typing import Optional

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from pi05_infer.dataconfig.libero_dataconfig import LeRobotLiberoDataConfig
from pi05_infer.dataconfig.turtle_dataconfig import TurtleDataConfig

_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi05_base"),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    TrainConfig(
        name="pi05_turtle",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50, discrete_state_input=False
        ),
        data=TurtleDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_base"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi05_base"),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def _override_with_data_kwargs(config: TrainConfig, data_kwargs: dict) -> TrainConfig:
    """Return a copy of the config with data_config set from openpi_data."""
    data_config = dataclasses.replace(config.data, **data_kwargs)
    replace_kwargs = {"data": data_config}
    return dataclasses.replace(config, **replace_kwargs)


def get_openpi_config(
    config_name: str,
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    repo_id: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
) -> TrainConfig:
    """Get a config by name.

    Args:
        config_name: Name of the config to load.
        model_path: Optional path to override model weights and assets.
        batch_size: Optional batch size override.
        repo_id: Optional LeRobot repo_id or local data path to override.
            When using a local path, the original asset_id is preserved so
            that norm_stats can still be loaded from the model checkpoint.
        data_kwargs: Optional overrides applied to the data config.
    """
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        config = _override_with_data_kwargs(config, data_kwargs)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    if repo_id is not None:
        original_repo_id = config.data.repo_id
        new_assets = dataclasses.replace(config.data.assets, asset_id=original_repo_id)
        new_data = dataclasses.replace(config.data, repo_id=repo_id, assets=new_assets)
        config = dataclasses.replace(config, data=new_data)

    return config
