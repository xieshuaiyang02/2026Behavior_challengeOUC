# """OUC JAX configs inherit the official baseline including filter, EMA and optimizer."""

# import dataclasses
# import json
# from pathlib import Path

# import tyro

# from openpi.models.pi0_config_ouc import Pi0ConfigOUC
# from openpi.training import config as original
# from openpi.training.stage_supervision_ouc import validate_stage_asset_config
# from openpi.training.weight_loaders import CheckpointWeightLoader
# from openpi.training.weight_loaders_ouc import StageCheckpointWeightLoader


# @dataclasses.dataclass(frozen=True)
# class TrainConfigOUC(original.TrainConfig):
#     model: Pi0ConfigOUC = dataclasses.field(
#         default_factory=lambda: Pi0ConfigOUC(num_stage_classes=2, action_horizon=32)
#     )
#     stage_assets_dir: str = "./outputs/assets/stage_annotations_8tasks_ouc"
#     dataset_root: str | None = None
#     use_task_stage_mask: bool = True


# def get_config(name):
#     names = {"pi05_b1k_ouc": "pi05_b1k", "pi05_b1k_8tasks_ouc": "pi05_b1k_8tasks"}
#     if name not in names:
#         raise ValueError(f"Unknown JAX OUC config: {name}")
#     base = original.get_config(names[name])
#     if not isinstance(base.weight_loader, CheckpointWeightLoader):
#         raise ValueError("This OUC profile expects the baseline's official CheckpointWeightLoader")
#     model = Pi0ConfigOUC(
#         **{f.name: getattr(base.model, f.name) for f in dataclasses.fields(base.model) if f.init}, num_stage_classes=2
#     )
#     data = dataclasses.replace(
#         base.data,
#         assets=dataclasses.replace(base.data.assets, assets_dir=base.data.assets.assets_dir or str(base.assets_dirs)),
#     )
#     values = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
#     values.update(
#         name=name,
#         model=model,
#         data=data,
#         weight_loader=StageCheckpointWeightLoader(base.weight_loader.params_path),
#         stage_assets_dir=str(
#             base.assets_dirs.parent / ("stage_annotations_8tasks_ouc" if "8tasks" in name else "stage_annotations_ouc")
#         ),
#     )
#     return TrainConfigOUC(**values)


# def resolve_stage_config(config):
#     root = Path(config.stage_assets_dir)
#     vocabulary = json.loads((root / "stage_vocab.json").read_text())
#     stage = json.loads((root / "stage_config.json").read_text())
#     validate_stage_asset_config(stage, vocabulary)
#     data = config.data
#     if config.dataset_root is not None:
#         data = dataclasses.replace(
#             data, base_config=dataclasses.replace(data.base_config, dataset_root=config.dataset_root)
#         )
#     return dataclasses.replace(
#         config, model=dataclasses.replace(config.model, num_stage_classes=len(vocabulary)), data=data
#     )


# def cli():
#     names = ["pi05_b1k_ouc"]
#     try:
#         original.get_config("pi05_b1k_8tasks")
#     except ValueError:
#         pass
#     else:
#         names.append("pi05_b1k_8tasks_ouc")
#     return tyro.extras.overridable_config_cli(
#         {name: ("Official JAX pi0.5 with OUC stage supervision", get_config(name)) for name in names}
#     )
































"""OUC JAX training configs.

This file follows the same registry style as OpenPI's original config.py:
    _CONFIGS -> _CONFIGS_DICT -> cli() / get_config()

Each OUC config can inherit an existing OpenPI baseline config while adding:
- OUC model heads
- stage supervision assets
- OUC-specific options

Add new training profiles by appending one entry to `_CONFIGS`.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
from pathlib import Path
from typing import Any

import tyro

from openpi.models.pi0_config_ouc import Pi0ConfigOUC
from openpi.training import config as original
from openpi.training.stage_supervision_ouc import validate_stage_asset_config
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training.weight_loaders_ouc import StageCheckpointWeightLoader


# ---------------------------------------------------------------------
# OUC TrainConfig
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainConfigOUC(original.TrainConfig):
    """OpenPI TrainConfig extended with OUC stage-supervision settings."""

    model: Pi0ConfigOUC = dataclasses.field(
        default_factory=lambda: Pi0ConfigOUC(
            num_stage_classes=2,
            action_horizon=32,
        )
    )

    # Placeholder/default. Normally each registered config overrides this.
    stage_assets_dir: str = "./outputs/assets/stage_annotations_ouc"

    # Allows CLI:
    #   --dataset-root /path/to/local/dataset
    dataset_root: str | None = None

    # Restrict classifier logits to stages valid for the current task.
    use_task_stage_mask: bool = True


# ---------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------


def _make_ouc_config(
    *,
    name: str,
    base_config: str,
    stage_assets_name: str | None = None,
    stage_assets_dir: str | None = None,
    use_task_stage_mask: bool = True,
    model_overrides: dict[str, Any] | None = None,
    **train_overrides: Any,
) -> TrainConfigOUC:
    """Create an OUC config by inheriting an existing OpenPI config.

    Args:
        name:
            Name exposed to CLI.

        base_config:
            Existing config name from openpi.training.config.

        stage_assets_name:
            Directory name under outputs/assets/.
            Ignored if stage_assets_dir is explicitly provided.

        stage_assets_dir:
            Explicit stage asset path.

        use_task_stage_mask:
            Whether to mask stage classes by task.

        model_overrides:
            Optional Pi0ConfigOUC field overrides.

        **train_overrides:
            Optional TrainConfig field overrides such as:
                batch_size=32
                fsdp_devices=4
                num_train_steps=30_000
                save_interval=5_000
    """

    base = original.get_config(base_config)

    if not isinstance(base.weight_loader, CheckpointWeightLoader):
        raise ValueError(
            f"OUC config {name!r} expects base config {base_config!r} "
            "to use CheckpointWeightLoader."
        )

    # -----------------------------------------------------------------
    # Convert original Pi0Config -> Pi0ConfigOUC
    # -----------------------------------------------------------------

    model_kwargs = {
        field.name: getattr(base.model, field.name)
        for field in dataclasses.fields(base.model)
        if field.init
    }

    # Temporary classifier size.
    # resolve_stage_config() replaces this with len(stage_vocab.json).
    model_kwargs["num_stage_classes"] = 2

    if model_overrides:
        model_kwargs.update(model_overrides)

    model = Pi0ConfigOUC(**model_kwargs)

    # -----------------------------------------------------------------
    # Keep baseline data config / norm-stat resolution
    # -----------------------------------------------------------------

    data = dataclasses.replace(
        base.data,
        assets=dataclasses.replace(
            base.data.assets,
            assets_dir=base.data.assets.assets_dir or str(base.assets_dirs),
        ),
    )

    # -----------------------------------------------------------------
    # Stage asset location
    # -----------------------------------------------------------------

    if stage_assets_dir is None:
        if stage_assets_name is None:
            stage_assets_name = f"stage_annotations_{name}"

        stage_assets_dir = str(
            base.assets_dirs.parent / stage_assets_name
        )

    # -----------------------------------------------------------------
    # Copy every original TrainConfig field
    # -----------------------------------------------------------------

    values = {
        field.name: getattr(base, field.name)
        for field in dataclasses.fields(base)
    }

    values.update(
        name=name,
        model=model,
        data=data,
        weight_loader=StageCheckpointWeightLoader(
            base.weight_loader.params_path
        ),
        stage_assets_dir=stage_assets_dir,
        use_task_stage_mask=use_task_stage_mask,
    )

    # Optional per-config overrides.
    values.update(train_overrides)

    return TrainConfigOUC(**values)


# =====================================================================
# OUC CONFIG REGISTRY
# =====================================================================
#
# Add new training configurations here.
#
# This follows the same idea as OpenPI's original:
#
#     _CONFIGS = [
#         TrainConfig(...),
#         TrainConfig(...),
#         ...
#     ]
#
# =====================================================================


_CONFIGS: list[TrainConfigOUC] = [

    # -----------------------------------------------------------------
    # BEHAVIOR-1K: turning_on_radio
    # -----------------------------------------------------------------

    _make_ouc_config(
        name="pi05_b1k_ouc",
        base_config="pi05_b1k",
        stage_assets_name="stage_annotations_ouc",
    ),

    # -----------------------------------------------------------------
    # BEHAVIOR-1K: selected 8 tasks
    # -----------------------------------------------------------------

    _make_ouc_config(
        name="pi05_b1k_8tasks_ouc",
        base_config="pi05_b1k_8tasks",
        stage_assets_name="stage_annotations_8tasks_ouc",
    ),

    # -----------------------------------------------------------------
    # Future configs
    # -----------------------------------------------------------------
    #
    # Example:
    #
    # _make_ouc_config(
    #     name="pi05_b1k_radio_ouc_test",
    #     base_config="pi05_b1k",
    #     stage_assets_dir="/absolute/path/to/stage_assets_ouc_radio",
    #     batch_size=32,
    #     fsdp_devices=4,
    #     num_train_steps=30_000,
    #     save_interval=5_000,
    # ),
    #
    # _make_ouc_config(
    #     name="pi05_b1k_8tasks_ouc_nomask",
    #     base_config="pi05_b1k_8tasks",
    #     stage_assets_name="stage_annotations_8tasks_ouc",
    #     use_task_stage_mask=False,
    # ),
    #
    # _make_ouc_config(
    #     name="pi05_b1k_8tasks_ouc_stage01",
    #     base_config="pi05_b1k_8tasks",
    #     stage_assets_name="stage_annotations_8tasks_ouc",
    #     model_overrides={
    #         "stage_loss_weight": 0.1,
    #         "next_stage_loss_weight": 0.1,
    #         "progress_loss_weight": 0.05,
    #     },
    # ),
]


# ---------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("OUC config names must be unique.")


_CONFIGS_DICT: dict[str, TrainConfigOUC] = {
    config.name: config
    for config in _CONFIGS
}


# ---------------------------------------------------------------------
# Stage asset resolution
# ---------------------------------------------------------------------


def resolve_stage_config(config: TrainConfigOUC) -> TrainConfigOUC:
    root = Path(config.stage_assets_dir)

    vocabulary = json.loads(
        (root / "stage_vocab.json").read_text()
    )
    stage = json.loads(
        (root / "stage_config.json").read_text()
    )

    validate_stage_asset_config(stage, vocabulary)

    # Exact training episodes come from OUC assets.
    manifest = json.loads(
        (root / "manifest_ouc.json").read_text()
    )

    episode_ids = sorted({
        int(ep["episode_id"])
        for ep in manifest["episodes"]
        if ep.get("split", "train") == "train"
    })

    if not episode_ids:
        raise ValueError(
            f"No training episodes found in {root / 'manifest_ouc.json'}"
        )

    base_config = config.data.base_config
    if base_config is None:
        raise ValueError(
            "B1K OUC config requires data.base_config"
        )

    dataset_kwargs = dict(base_config.dataset_kwargs or {})
    dataset_kwargs["episodes"] = episode_ids

    base_config = dataclasses.replace(
        base_config,
        dataset_root=(
            config.dataset_root
            if config.dataset_root is not None
            else base_config.dataset_root
        ),
        dataset_kwargs=dataset_kwargs,
    )

    data = dataclasses.replace(
        config.data,
        base_config=base_config,
    )

    return dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            num_stage_classes=len(vocabulary),
        ),
        data=data,
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def get_config(config_name: str) -> TrainConfigOUC:
    """Get an OUC training config by name."""

    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name,
            _CONFIGS_DICT.keys(),
            n=1,
            cutoff=0.0,
        )

        closest_str = (
            f" Did you mean {closest[0]!r}?"
            if closest
            else ""
        )

        raise ValueError(
            f"OUC config {config_name!r} not found."
            f"{closest_str}"
        )

    return _CONFIGS_DICT[config_name]


def cli() -> TrainConfigOUC:
    """OUC command-line config selector."""

    return tyro.extras.overridable_config_cli(
        {
            name: (
                f"OUC training config: {name}",
                config,
            )
            for name, config in _CONFIGS_DICT.items()
        }
    )