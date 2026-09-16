"""Official JAX checkpoint loading with narrowly allowed new OUC parameters."""

import dataclasses

import flax.traverse_util
import numpy as np

from openpi.models import model
from openpi.shared import download
from openpi.training.weight_loaders import _merge_params

STAGE_ROOTS = {"stage_queries", "current_stage_head", "next_stage_head", "progress_head"}


@dataclasses.dataclass(frozen=True)
class StageCheckpointWeightLoader:
    params_path: str

    def load(self, params):
        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        expected = flax.traverse_util.flatten_dict(params, sep="/")
        actual = flax.traverse_util.flatten_dict(loaded, sep="/")
        missing = set(expected) - set(actual)
        invalid = {key for key in missing if key.split("/")[0] not in STAGE_ROOTS and "lora" not in key}
        if invalid or set(actual) - set(expected):
            raise ValueError(
                f"Incompatible official checkpoint: missing={sorted(invalid)}, unexpected={sorted(set(actual) - set(expected))}"
            )
        stage_keys = {key for key in expected if key.split("/")[0] in STAGE_ROOTS}
        if missing & stage_keys and missing & stage_keys != stage_keys:
            raise ValueError(
                "Partially initialized stage weights; use the official base checkpoint or a complete OUC checkpoint"
            )
        return _merge_params(
            loaded,
            params,
            missing_regex=r"(?:stage_queries|current_stage_head|next_stage_head|progress_head)(?:/.*)?|.*lora.*",
        )
