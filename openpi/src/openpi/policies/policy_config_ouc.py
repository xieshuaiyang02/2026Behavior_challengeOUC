"""Restore OUC architecture from assets and delegate to the official JAX policy loader."""

import dataclasses
import json
from pathlib import Path

from openpi.models.pi0_config_ouc import Pi0ConfigOUC
from openpi.policies import policy_config as original


def create_trained_policy(train_config, checkpoint_dir, **kwargs):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "params").is_dir() or (checkpoint_dir / "model.safetensors").exists():
        raise ValueError("Expected an OUC JAX checkpoint step directory containing params/")
    root = checkpoint_dir / "assets"
    values = json.loads((root / "model_config_ouc.json").read_text())
    vocabulary = json.loads((root / "stage_vocab.json").read_text())
    if vocabulary.get("<DONE>") != 0 or sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Invalid OUC stage vocabulary")
    if values["num_stage_classes"] != len(vocabulary):
        raise ValueError("Checkpoint model and vocabulary dimensions disagree")
    model = Pi0ConfigOUC(**values)
    metadata = dict(train_config.policy_metadata or {})
    metadata.update(action_horizon=model.action_horizon, stage_aware=True, backend="jax")
    config = dataclasses.replace(train_config, model=model, policy_metadata=metadata)
    return original.create_trained_policy(config, checkpoint_dir, **kwargs)
