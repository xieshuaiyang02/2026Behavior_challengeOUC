"""Official Orbax/EMA checkpoint layout plus OUC semantic assets in the asset callback."""

import dataclasses
import hashlib
import json
from pathlib import Path

from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import checkpoints as original
from openpi.training.stage_supervision_ouc import TRAINING_TARGET_POLICY

initialize_checkpoint_dir = original.initialize_checkpoint_dir
ASSETS = ("stage_vocab.json", "task_stage_vocab.json", "stage_config.json", "stage_id_to_metadata.json")
IDENTITY_ASSETS = (*ASSETS, "stage_segments.parquet", "episode_stage_metadata.json")


def asset_fingerprints(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in IDENTITY_ASSETS}


def save_state(checkpoint_manager, state, data_loader, step):
    # Snapshot metadata now so asynchronous writes cannot observe changed assets.
    root = Path(data_loader.stage_assets_dir)
    assets = {name: (root / name).read_text() for name in ASSETS}
    fingerprint = asset_fingerprints(root)
    model_config = dataclasses.asdict(data_loader.stage_model_config)
    target_policy = {
        "policy": TRAINING_TARGET_POLICY,
        "task_stage_mask": getattr(data_loader, "use_task_stage_mask", True),
        "loss_masks": "current_ce_and_progress=current_stage_valid;next_ce=next_stage_valid",
    }
    data_config = data_loader.data_config()

    def save_assets(directory):
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
            normalize.save(directory / asset_id, data_config.norm_stats)
        for name, value in assets.items():
            (directory / name).write_text(value)
        (directory / "training_target_policy_ouc.json").write_text(json.dumps(target_policy, indent=2))
        (directory / "model_config_ouc.json").write_text(json.dumps(model_config, indent=2))
        (directory / "stage_asset_fingerprints_ouc.json").write_text(json.dumps(fingerprint, indent=2))

    with at.disable_typechecking():
        train_state, params = original._split_params(state)
    checkpoint_manager.save(step, {"assets": save_assets, "train_state": train_state, "params": {"params": params}})


def restore_state(checkpoint_manager, state, data_loader, step=None):
    step = checkpoint_manager.latest_step() if step is None else step
    directory = Path(str(checkpoint_manager.directory)) / str(step) / "assets"
    policy_path = directory / "training_target_policy_ouc.json"
    if not policy_path.is_file():
        raise ValueError("Checkpoint predates independent loss masks/contiguous-next policy; start a new experiment")
    policy = json.loads(policy_path.read_text())
    if policy.get("policy") != TRAINING_TARGET_POLICY or policy.get("task_stage_mask") != getattr(
        data_loader, "use_task_stage_mask", True
    ):
        raise ValueError("OUC training target policy changed since checkpoint")
    expected = json.loads((directory / "stage_asset_fingerprints_ouc.json").read_text())
    if expected != asset_fingerprints(data_loader.stage_assets_dir):
        raise ValueError("Stage assets changed since checkpoint; cannot resume with different labels or class IDs")
    saved_model = json.loads((directory / "model_config_ouc.json").read_text())
    if saved_model != dataclasses.asdict(data_loader.stage_model_config):
        raise ValueError("OUC model configuration changed since checkpoint")
    return original.restore_state(checkpoint_manager, state, data_loader, step=step)
