"""OUC stage targets on the official B1K JAX-sharded anchor/chunk loader."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator
import dataclasses
from itertools import accumulate
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training.stage_annotations_ouc import IGNORE_ID
from openpi.training.stage_supervision_ouc import StageSupervisionLookup

_TARGET_KEYS = frozenset(
    {
        "current_stage_id",
        "next_stage_id",
        "progress",
        "stage_progress",
        "stage_valid",
        "current_stage_valid",
        "next_stage_valid",
        "valid_mask",
        "task_id",
        "allowed_stage_mask",
        "stage_targets",
    }
)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def validate_stage_targets(targets, num_classes):
    """Validate scalar lookup output on the host, before any JAX CE/gather."""
    required = {
        "current_stage_id",
        "next_stage_id",
        "progress",
        "current_stage_valid",
        "next_stage_valid",
        "stage_valid",
        "task_id",
    }
    if missing := required.difference(targets):
        raise ValueError(f"Schema 2 lookup is missing target fields: {sorted(missing)}")
    current_valid, next_valid = bool(targets["current_stage_valid"]), bool(targets["next_stage_valid"])
    if bool(targets["stage_valid"]) != (current_valid and next_valid):
        raise ValueError("stage_valid must equal current_stage_valid AND next_stage_valid")
    for key, valid, minimum in (("current_stage_id", current_valid, 1), ("next_stage_id", next_valid, 0)):
        label = targets[key]
        if not isinstance(label, (int, np.integer)) or isinstance(label, (bool, np.bool_)):
            raise ValueError(f"{key} must be an integer")
        if valid and not minimum <= label < num_classes:
            raise ValueError(f"Invalid supervised {key}={label}")
        if not valid and label != IGNORE_ID:
            raise ValueError(f"Invalid {key} must use IGNORE_ID={IGNORE_ID}")
        if valid and label > 0 and "allowed_stage_mask" in targets:
            mask = np.asarray(targets["allowed_stage_mask"])
            if mask.shape != (num_classes,) or not mask[label]:
                raise ValueError(f"Task mask excludes supervised {key}={label}")
    if current_valid and (not np.isfinite(targets["progress"]) or not 0 <= targets["progress"] <= 1):
        raise ValueError("Supervised progress must be finite and in [0, 1]")


class RepositoryDataset:
    """Concatenate separately constructed repositories while preserving local episode IDs."""

    def __init__(self, datasets: list[Any]):
        self._datasets = datasets
        self._ends = list(accumulate(len(dataset) for dataset in datasets))

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        repository = bisect_right(self._ends, index)
        offset = self._ends[repository - 1] if repository else 0
        return {**self._datasets[repository][index - offset], "dataset_index": repository}


def create_b1k_dataset(data_config: Any, action_horizon: int) -> Any:
    """Use each repository's own fps and task prompt mapping when loading many tasks."""
    from openpi.training import data_loader as original  # noqa: PLC0415
    from openpi.training import lerobot_compat  # noqa: PLC0415

    if not isinstance(data_config.repo_id, list):
        return original.create_b1k_dataset(data_config, action_horizon)
    if not data_config.repo_id or data_config.dataset_root is None:
        raise ValueError("Multiple BEHAVIOR repositories require repo IDs and their shared dataset_root")
    data_cls = data_config.data_cls
    if data_cls is lerobot_compat.MultiLeRobotDataset:
        data_cls = lerobot_compat.LeRobotDataset
    datasets = []
    for repo_id in data_config.repo_id:
        per_repo_config = dataclasses.replace(
            data_config,
            repo_id=repo_id,
            dataset_root=str(Path(data_config.dataset_root) / repo_id),
            data_cls=data_cls,
        )
        datasets.append(original.create_b1k_dataset(per_repo_config, action_horizon))
    return RepositoryDataset(datasets)


class StageJoinedDataset:
    """Join the raw anchor once, then transform only observation/action data.

    The action chunk is read by the original dataset and passed to exactly the
    same repack, action, normalization and model transformations as before.
    Labels live in a separate tuple branch, so repack transforms cannot drop
    them and prompt transforms cannot see them.
    """

    def __init__(
        self,
        dataset: Any,
        lookup: StageSupervisionLookup,
        transform: Callable[[dict], dict],
        *,
        require_dataset_index: bool = False,
    ):
        self._dataset = dataset
        self._lookup = lookup
        self._transform = transform
        self._require_dataset_index = require_dataset_index

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        sample = dict(self._dataset[index])
        episode = sample.get("episode_id", sample.get("episode_index"))
        if episode is None:
            raise KeyError("OUC stage join requires raw episode_id or episode_index before repacking")
        if self._require_dataset_index and "dataset_index" not in sample:
            raise KeyError("Multiple repositories require a raw dataset_index to disambiguate episode IDs")
        frame = sample.get("frame_index", sample.get("frame_idx"))
        timestamp = sample.get("timestamp")
        source_frame = sample.get("source_frame_index", sample.get("source_frame_idx"))
        targets = self._lookup.lookup(
            _scalar(episode),
            _scalar(frame),
            timestamp=_scalar(timestamp),
            source_frame_index=_scalar(source_frame),
            dataset_index=_scalar(sample.get("dataset_index", 0)),
        )
        validate_stage_targets(targets, self._lookup.num_stage_classes)
        model_data = self._transform({key: value for key, value in sample.items() if key not in _TARGET_KEYS})
        if _TARGET_KEYS.intersection(model_data):
            raise ValueError("Data transforms must not introduce stage target fields into model observations")
        return model_data, targets


class StageDataLoader:
    def __init__(self, data_config: Any, data_loader: Any):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> Any:
        return self._data_config

    def __iter__(self) -> Iterator[tuple[Any, Any, dict]]:
        from openpi.models.model import Observation  # noqa: PLC0415

        for batch, targets in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"], targets


def create_data_loader(config, *, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False):
    import json

    import jax

    from openpi.training import data_loader as original
    import openpi.transforms as transforms

    data = config.data.create(config.assets_dirs, config.model)
    if data.rlds_data_dir is not None or data.repo_id == "fake":
        raise ValueError("OUC requires a real map-style LeRobot dataset with episode/frame identities")
    lookup = StageSupervisionLookup.from_assets(config.stage_assets_dir, use_task_stage_mask=config.use_task_stage_mask)
    if lookup.num_stage_classes != config.model.num_stage_classes:
        raise ValueError("Model classifier size disagrees with stage vocabulary")
    dataset = (
        create_b1k_dataset(data, config.model.action_horizon)
        if data.dataset_root is not None
        else original.create_torch_dataset(data, config.model.action_horizon, config.model)
    )
    # Check global episode IDs even when assets have no selection_audit_ouc.json.
    raw = dataset
    while hasattr(raw, "_dataset"):
        raw = raw._dataset
    if hasattr(raw, "hf_dataset") and not isinstance(data.repo_id, list):
        actual = sorted(int(value) for value in raw.hf_dataset.unique("episode_index"))
        expected = sorted(int(e["episode_id"]) for e in lookup._episodes.values() if e["dataset_index"] == 0)
        frames = sum(e["dataset_num_frames"] for e in lookup._episodes.values())
        if actual != expected or len(dataset) != frames:
            raise ValueError("Training episode IDs/frame count differ from stage assets; select matching data/assets")
    audit_file = Path(config.stage_assets_dir) / "selection_audit_ouc.json"
    if audit_file.is_file():
        audit = json.loads(audit_file.read_text())
        if not hasattr(raw, "hf_dataset"):
            raise TypeError("Cannot verify the prepared baseline selection against the current dataset")
        actual = sorted(int(value) for value in raw.hf_dataset.unique("episode_index"))
        if actual != audit["selected_episode_ids"] or len(dataset) != audit["num_frames"]:
            raise ValueError("Training dataset differs from stage selection audit; use matching data/assets")
    if not skip_norm_stats and data.norm_stats is None:
        raise ValueError("Existing baseline normalization stats not found; check data.assets.assets_dir / asset_id")
    transform = transforms.compose(
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            transforms.Normalize({} if skip_norm_stats else data.norm_stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    dataset = StageJoinedDataset(dataset, lookup, transform, require_dataset_index=isinstance(data.repo_id, list))
    batches = original.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )
    loader = StageDataLoader(data, batches)
    loader.stage_assets_dir = config.stage_assets_dir
    loader.stage_model_config = config.model
    loader.use_task_stage_mask = config.use_task_stage_mask
    loader.target_policy_report = lookup.target_policy_report
    return loader
