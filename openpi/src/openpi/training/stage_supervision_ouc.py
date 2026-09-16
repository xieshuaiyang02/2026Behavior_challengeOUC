"""Training-only policy on top of the verified server schema 2 annotation API.

The builder and stage_annotations_ouc.py stay identical to the supplied server
files. Load existing assets through their StageAnnotationLookup.from_assets;
only private in-memory copies of unsafe next IDs are masked for training.
"""

from collections.abc import Mapping
import json
import logging
from pathlib import Path

from openpi.training.stage_annotations_ouc import DONE_ID
from openpi.training.stage_annotations_ouc import DONE_KEY
from openpi.training.stage_annotations_ouc import IGNORE_ID
from openpi.training.stage_annotations_ouc import SCHEMA_VERSION
from openpi.training.stage_annotations_ouc import StageAnnotationLookup

TRAINING_TARGET_POLICY = "schema2_contiguous_next_independent_valid_v1"


def validate_stage_asset_config(config: Mapping, vocabulary: Mapping) -> None:
    """Check the actual server schema's coordinate and label conventions."""
    if (
        SCHEMA_VERSION != 2
        or config.get("schema_version") != SCHEMA_VERSION
        or config.get("coordinate_system") != "source_frames"
        or config.get("interval_convention") != "half_open"
        or config.get("unknown_stage_id") != IGNORE_ID
        or config.get("done_stage_id") != DONE_ID
        or config.get("num_stage_queries") != 3
        or config.get("num_stage_classes") != len(vocabulary)
        or vocabulary.get(DONE_KEY) != DONE_ID
        or sorted(vocabulary.values()) != list(range(len(vocabulary)))
    ):
        raise ValueError("Incompatible schema 2 OUC assets: check IDs, vocabulary, queries and source-frame policy")


class StageSupervisionLookup(StageAnnotationLookup):
    """Reuse server lookup/time mapping, adding explicit training-only next guards."""

    def __init__(
        self, segments, episode_metadata, stage_vocab, task_stage_vocab, *, split="train", use_task_stage_mask=True
    ):
        super().__init__(
            segments,
            episode_metadata,
            stage_vocab,
            task_stage_vocab,
            split=split,
            use_task_stage_mask=use_task_stage_mask,
        )
        # The supplied server constructor copies each row before populating _rows.
        # We retain the source next ID for audit and never write back to assets.
        self.target_policy_report = {"policy": TRAINING_TARGET_POLICY, "masked_next_segments": 0}
        for key, rows in self._rows.items():
            for index, row in enumerate(rows):
                if self.use_task_stage_mask and not row.get("is_ambiguous", False):
                    for id_key in ("current_stage_id", "next_stage_id"):
                        stage_id = row[id_key]
                        if stage_id > 0 and not self._task_masks[row["task_id"]][stage_id]:
                            raise ValueError(f"Task mask excludes {id_key}={stage_id} in episode {key}")
                following = rows[index + 1] if index + 1 < len(rows) else None
                if following is None:
                    safe = bool(row.get("is_terminal_stage", False)) and row["next_stage_id"] == DONE_ID
                else:
                    safe = (
                        row["end_frame"] == following["start_frame"]
                        and not following.get("is_ambiguous", False)
                        and following["current_stage_id"] > 0
                        and row["next_stage_id"] == following["current_stage_id"]
                    )
                safe = safe and not row.get("is_ambiguous", False) and row["current_stage_id"] > 0
                if row["next_stage_id"] != IGNORE_ID and not safe:
                    row["stored_next_stage_id"] = row["next_stage_id"]
                    row["next_stage_id"] = IGNORE_ID
                    self.target_policy_report["masked_next_segments"] += 1
        logging.info("OUC training target policy: %s", self.target_policy_report)
        if self.target_policy_report["masked_next_segments"]:
            logging.warning(
                "Unsafe stored next-stage targets are masked for training; source assets unchanged: %s",
                self.target_policy_report,
            )

    @classmethod
    def from_assets(cls, stage_assets_dir, *, split="train", use_task_stage_mask=True):
        directory = Path(stage_assets_dir)
        config = json.loads((directory / "stage_config.json").read_text())
        vocabulary = json.loads((directory / "stage_vocab.json").read_text())
        validate_stage_asset_config(config, vocabulary)
        # The server factory constructs cls, so the above training guard is applied.
        return super().from_assets(directory, split=split, use_task_stage_mask=use_task_stage_mask)

    # lookup() is inherited verbatim: timestamp/source-frame mapping, current
    # and next validity, and original progress bounds are owned by the server API.
