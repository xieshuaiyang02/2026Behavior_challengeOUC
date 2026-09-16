"""OUC semantic stage assets and anchor lookup, independent of the model.

Segments always use the annotation's source frame coordinates. A dataset anchor
is mapped into that coordinate system before lookup; frame indices alone are
accepted only when the manifest explicitly declares equal source/dataset rates.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

DONE_KEY = "<DONE>"
DONE_ID = 0
IGNORE_ID = -100
SCHEMA_VERSION = 2


def _token(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a nonempty semantic string, got {value!r}")
    return re.sub(r"\s+", "_", value.strip().lower())


def normalize_object_id(object_instance_id: str, episode_metadata: Mapping | None = None) -> str:
    """Resolve metadata categories first, then strip known BEHAVIOR instance suffixes.

    Supported metadata: direct instance->category/record maps, an
    ``object_categories``/``object_map`` map, or ``objects`` map/list of records
    containing ``name`` (or ``instance_id``) and ``category``/``object_category``.
    The conservative fallback recognizes six-character model IDs and numeric
    instance IDs; callers should supply metadata for other naming conventions.
    """
    instance = object_instance_id
    _token(instance)  # Validate before map lookup or regex operations.
    metadata = episode_metadata or {}
    candidates = [metadata]
    candidates.extend(metadata.get(key, {}) for key in ("object_categories", "object_map", "objects"))
    for source in candidates:
        candidate = source
        if isinstance(source, list):
            candidate = {row.get("instance_id", row.get("name")): row for row in source if isinstance(row, Mapping)}
        if not isinstance(candidate, Mapping) or instance not in candidate:
            continue
        record = candidate[instance]
        category = record.get("object_category", record.get("category")) if isinstance(record, Mapping) else record
        if category is not None:
            return _token(category)
    instance = re.sub(r"_[a-zA-Z0-9]{6}_\d+$", "", instance)
    instance = re.sub(r"_\d+$", "", instance)
    return _token(instance)


def _strings(annotation: Mapping, key: str) -> list[str]:
    values = annotation.get(key, [])
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a list of strings")
    def _flatten(items):
        if isinstance(items, str):
            if not items.strip():
                return []
            return [_token(items)]
        if isinstance(items, (list, tuple)):
            return [
                token
                for item in items
                for token in _flatten(item)
            ]
        raise ValueError(
            f"Expected a nonempty semantic string, got {items!r}"
        )

    return _flatten(values)


def build_stage_key(skill_ann: Mapping, object_map: Mapping | None = None) -> str:
    """Collision-free canonical serialization preserving ordered object groups."""
    skills = _strings(skill_ann, "skill_description")
    if not skills:
        raise ValueError("skill_description must contain at least one skill")
    groups = skill_ann.get("object_id", [])
    if not isinstance(groups, (list, tuple)) or any(not isinstance(group, (list, tuple)) for group in groups):
        raise ValueError("object_id must be a list of object groups")
    def _object_members(items):
        for value in items:
            if isinstance(value, str):
                if value.strip():
                    yield value
            elif isinstance(value, (list, tuple)):
                yield from _object_members(value)
            else:
                raise ValueError(f"Invalid object_id member: {value!r}")
    objects = [
        [normalize_object_id(value, object_map) for value in _object_members(group)]
        for group in groups
    ]
    return json.dumps(
        [skills, objects, _strings(skill_ann, "memory_prefix"), _strings(skill_ann, "spatial_prefix")],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stage_text(stage_key: str) -> str:
    """Human-readable debug text; never tokenized as a model input."""
    if stage_key == DONE_KEY:
        return DONE_KEY
    skills, groups, memory, spatial = json.loads(stage_key)
    parts = [" / ".join(skills), "; ".join(", ".join(group) for group in groups)]
    if memory:
        parts.append("memory=" + ", ".join(memory))
    if spatial:
        parts.append("spatial=" + ", ".join(spatial))
    return " ".join(part for part in parts if part).replace("_", " ")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _episode_key(episode_id: Any, dataset_index: int = 0) -> str:
    return json.dumps([int(dataset_index), str(episode_id)], separators=(",", ":"))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# OUC_OVERLAP_POLICY_V2
# OUC_SOURCE_BOUNDS_POLICY_V2_1
def _ouc_parse_intervals(duration, episode_key, skill_idx, source_frames):
    """Return bounded intervals, original bounds, and invalidity metadata."""
    if (
        isinstance(duration, (list, tuple))
        and len(duration) == 2
        and not any(isinstance(x, (list, tuple)) for x in duration)
    ):
        intervals = [duration]
    elif (
        isinstance(duration, (list, tuple))
        and duration
        and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in duration)
    ):
        intervals = duration
    else:
        raise ValueError(
            f"Invalid frame_duration for {episode_key}, skill {skill_idx}: {duration!r}"
        )
    result = []
    for interval_idx, interval in enumerate(intervals):
        try:
            raw_start, raw_end = (_integer(x, "frame_duration") for x in interval)
        except ValueError as exc:
            raise ValueError(
                f"Invalid frame_duration for {episode_key}, skill {skill_idx}, "
                f"interval {interval_idx}: {interval!r}"
            ) from exc
        if raw_start == raw_end:
            raise ValueError(f"Empty frame_duration: {episode_key}, {interval!r}")
        reversed_bounds = raw_start > raw_end
        start = min(raw_start, raw_end)
        end = min(max(raw_start, raw_end), source_frames)
        if not 0 <= start < end <= source_frames:
            raise ValueError(
                f"Interval has no usable source-frame span: {episode_key}, {interval!r}"
            )
        result.append({
            "start_frame": start,
            "end_frame": end,
            "raw_start_frame": raw_start,
            "raw_end_frame": raw_end,
            "is_invalid": reversed_bounds,
            "invalid_reason": "reversed_bounds" if reversed_bounds else None,
            "source_interval_adjustment": (
                "clip_to_source_end" if max(raw_start, raw_end) > source_frames else None
            ),
        })
    return result

def _ouc_exclusive_rows(raw_rows, episode_key):
    """Partition raw intervals; multiple active sources are ambiguous.

    Do not infer simultaneous execution, merge directions, or choose a winner.
    Preserve original progress bounds and source interval identities.
    """
    if not raw_rows:
        return [], []
    from collections import defaultdict

    by_id = {row["raw_segment_id"]: row for row in raw_rows}
    if len(by_id) != len(raw_rows):
        raise ValueError(f"Duplicate raw segment identity in {episode_key}")
    starts, ends = defaultdict(set), defaultdict(set)
    for row in raw_rows:
        ident = row["raw_segment_id"]
        starts[row["start_frame"]].add(ident)
        ends[row["end_frame"]].add(ident)
    boundaries = sorted(set(starts) | set(ends))
    active = set()
    result = []
    common = ("episode_id", "dataset_index", "split", "task_id", "task_name")
    for left, right in zip(boundaries, boundaries[1:]):
        active.difference_update(ends[left])
        active.update(starts[left])
        if not active or left >= right:
            continue
        ids = tuple(sorted(active))
        if result and result[-1]["end_frame"] == left and tuple(result[-1]["raw_segment_ids"]) == ids:
            result[-1]["end_frame"] = right
            continue
        source = by_id[ids[0]]
        row = {key: source[key] for key in common}
        row.update(
            start_frame=left,
            end_frame=right,
            raw_segment_ids=list(ids),
            is_ambiguous=(len(ids) != 1 or any(by_id[ident].get("is_invalid", False) for ident in ids)),
        )
        if not row["is_ambiguous"]:
            row.update(
                skill_idx=source["skill_idx"],
                interval_idx=source["interval_idx"],
                progress_start_frame=source["start_frame"],
                progress_end_frame=source["end_frame"],
                current_stage_key=source["current_stage_key"],
                current_stage_text=source["current_stage_text"],
            )
        else:
            row.update(
                skill_idx=-1,
                interval_idx=-1,
                progress_start_frame=left,
                progress_end_frame=right,
                current_stage_key=None,
                current_stage_text="",
            )
        result.append(row)
    conflicts = []
    for row in result:
        if not row["is_ambiguous"]:
            continue
        sources = [by_id[ident] for ident in row["raw_segment_ids"]]
        conflicts.append({
            "episode_key": episode_key,
            "episode_id": row["episode_id"],
            "task_id": row["task_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "raw_segment_ids": row["raw_segment_ids"],
            "skill_indices": [s["skill_idx"] for s in sources],
            "stage_keys": [s["current_stage_key"] for s in sources],
            "reason": "invalid_source_interval" if any(s.get("is_invalid", False) for s in sources) else "overlapping_intervals",
        })
    for segment_idx, row in enumerate(result):
        row["segment_idx"] = segment_idx
    return result, conflicts


def build_stage_assets(manifest_path: str | Path, output_dir: str | Path) -> dict:
    """Build a train-only vocabulary and segment-level parquet from a JSON manifest.

    Each episode requires episode_id, task_id, task_name, split, source_fps,
    source_num_frames, dataset_fps, dataset_num_frames, and annotation_path (or
    an inline annotation mapping). Paths resolve relative to the manifest.
    Optional dataset_index disambiguates equal episode IDs across repositories.
    Optional timestamp_offset defaults to zero, in seconds on the source clock.
    Unknown validation/test stages retain their canonical keys but get -100 IDs.
    """
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = _read_json(manifest_path)
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest must contain a nonempty episodes list")

    metadata, pending, raw_pending, conflicts, train_keys = {}, [], [], [], set()
    task_names, task_ids = {}, {}
    train_tasks: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        dataset_index = _integer(episode.get("dataset_index", 0), "dataset_index")
        key = _episode_key(episode_id, dataset_index)
        if key in metadata:
            raise ValueError(f"Duplicate episode identity: {key}")
        task_id = _integer(episode["task_id"], "task_id")
        task_name = episode["task_name"]
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a nonempty string")
        if task_id in task_names and task_names[task_id] != task_name:
            raise ValueError(f"Task ID {task_id} maps to multiple names")
        if task_name in task_ids and task_ids[task_name] != task_id:
            raise ValueError(f"Task name {task_name!r} maps to multiple IDs")
        task_names[task_id], task_ids[task_name] = task_name, task_id
        split = episode["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        source_frames = _integer(episode["source_num_frames"], "source_num_frames", minimum=1)
        source_fps = _positive(episode["source_fps"], "source_fps")
        dataset_frames = _integer(episode["dataset_num_frames"], "dataset_num_frames", minimum=1)
        dataset_fps = _positive(episode["dataset_fps"], "dataset_fps")
        offset = float(episode.get("timestamp_offset", 0.0))
        if not math.isfinite(offset) or offset < 0 or offset * source_fps >= source_frames:
            raise ValueError("timestamp_offset must lie inside the source episode")
        # Nominal rate mapping is also checked at runtime against actual timestamps.
        if ((dataset_frames - 1) / dataset_fps + offset) * source_fps >= source_frames + 1e-4:
            raise ValueError(f"Dataset duration exceeds source episode bounds: {key}")
        metadata[key] = {
            "episode_id": episode_id,
            "dataset_index": dataset_index,
            "task_id": task_id,
            "task_name": task_name,
            "split": split,
            "source_fps": source_fps,
            "source_num_frames": source_frames,
            "dataset_fps": dataset_fps,
            "dataset_num_frames": dataset_frames,
            "timestamp_offset": offset,
        }
        if "annotation" in episode:
            annotation = episode["annotation"]
        else:
            annotation = _read_json(manifest_path.parent / episode["annotation_path"])
        if not isinstance(annotation, Mapping) or "skill_annotation" not in annotation:
            raise ValueError(f"Annotation for {key} must contain skill_annotation (use [] for unlabeled episodes)")
        skills = annotation["skill_annotation"]
        if not isinstance(skills, list):
            raise ValueError("skill_annotation must be a list")
        object_metadata = episode.get("object_metadata", {})
        if "object_metadata_path" in episode:
            object_metadata = _read_json(manifest_path.parent / episode["object_metadata_path"])
        raw_rows, seen_indices = [], set()
        for index, skill in enumerate(skills):
            skill_idx = _integer(skill.get("skill_idx", index), "skill_idx")
            if skill_idx in seen_indices:
                raise ValueError(f"Duplicate skill_idx {skill_idx} in episode {key}")
            seen_indices.add(skill_idx)
            stage_key = build_stage_key(skill, object_metadata)
            intervals = _ouc_parse_intervals(
                skill.get("frame_duration"), key, skill_idx, source_frames
            )
            if split == "train":
                train_keys.add(stage_key)
                train_tasks[task_name].add(stage_key)
            for interval_idx, interval in enumerate(intervals):
                raw_rows.append({
                    "episode_id": episode_id,
                    "dataset_index": dataset_index,
                    "split": split,
                    "task_id": task_id,
                    "task_name": task_name,
                    "skill_idx": skill_idx,
                    "interval_idx": interval_idx,
                    "raw_segment_id": f"{skill_idx}:{interval_idx}",
                    **interval,
                    "current_stage_key": stage_key,
                    "current_stage_text": stage_text(stage_key),
                })
        rows, episode_conflicts = _ouc_exclusive_rows(raw_rows, key)
        for index, row in enumerate(rows):
            following = rows[index + 1] if index + 1 < len(rows) else None
            # A conflict is a barrier: do not skip it to invent a next stage.
            next_key = following["current_stage_key"] if following is not None else DONE_KEY
            if row["is_ambiguous"]:
                next_key = None
            row["next_stage_key"] = next_key
            row["next_stage_text"] = stage_text(next_key) if next_key is not None else ""
            row["is_terminal_stage"] = following is None and not row["is_ambiguous"]
        raw_pending.extend(raw_rows)
        conflicts.extend(episode_conflicts)
        pending.extend(rows)

    if not train_keys:
        raise ValueError("At least one training skill is needed to build the vocabulary")
    vocabulary = {DONE_KEY: DONE_ID, **{key: index for index, key in enumerate(sorted(train_keys), start=1)}}
    for row in pending:
        row["current_stage_id"] = vocabulary.get(row["current_stage_key"], IGNORE_ID)
        row["next_stage_id"] = vocabulary.get(row["next_stage_key"], IGNORE_ID)
    pending.sort(key=lambda row: (row["dataset_index"], row["episode_id"], row["start_frame"]))
    task_vocabulary = {name: sorted(vocabulary[key] for key in keys) for name, keys in sorted(train_tasks.items())}
    reverse = {str(index): {"stage_key": key, "stage_text": stage_text(key)} for key, index in vocabulary.items()}
    config = {
        "schema_version": SCHEMA_VERSION,
        "num_stage_classes": len(vocabulary),
        "done_stage_id": DONE_ID,
        "num_stage_queries": 3,
        "coordinate_system": "source_frames",
        "interval_convention": "half_open",
        "progress_name": "temporal_stage_progress",
        "vocabulary_split": "train",
        "unknown_stage_id": IGNORE_ID,
        "overlap_policy": "mask_multiple_raw_intervals",
        "progress_bounds": "original_source_interval",
        "source_bounds_policy": "v2.1_mask_reversed_clip_end",
        "num_invalid_raw_intervals": sum(r["is_invalid"] for r in raw_pending),
        "num_clipped_raw_intervals": sum(r["source_interval_adjustment"] is not None for r in raw_pending),
        "num_conflict_intervals": len(conflicts),
        "num_conflict_frames": sum(c["end_frame"] - c["start_frame"] for c in conflicts),
    }
    # Import before writing any assets so missing parquet support fails cleanly.
    import polars as pl  # noqa: PLC0415

    table = pl.DataFrame(
        pending,
        infer_schema_length=None,
        schema_overrides={
            "current_stage_key": pl.String,
            "next_stage_key": pl.String,
        },
    )
    raw_table = pl.DataFrame(
        raw_pending,
        infer_schema_length=None,
        schema_overrides={
            "invalid_reason": pl.String,
            "source_interval_adjustment": pl.String,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_dir / "stage_segments.parquet")
    raw_table.write_parquet(output_dir / "raw_stage_segments.parquet")
    _write_json(output_dir / "stage_conflicts.json", conflicts)
    _write_json(output_dir / "stage_invalid_intervals.json", [
        r for r in raw_pending if r["is_invalid"] or r["source_interval_adjustment"] is not None
    ])
    _write_json(output_dir / "stage_vocab.json", vocabulary)
    _write_json(output_dir / "task_stage_vocab.json", task_vocabulary)
    _write_json(output_dir / "stage_id_to_metadata.json", reverse)
    _write_json(output_dir / "episode_stage_metadata.json", metadata)
    _write_json(output_dir / "stage_config.json", config)
    return {**config, "num_episodes": len(metadata), "num_segments": len(pending), "num_raw_segments": len(raw_pending), "num_valid_segments": sum(not r["is_ambiguous"] for r in pending)}


class StageAnnotationLookup:
    """Binary search over validated source-frame segments; gaps stay unlabeled."""

    def __init__(
        self,
        segments: list[dict],
        episode_metadata: Mapping,
        stage_vocab: Mapping,
        task_stage_vocab: Mapping,
        *,
        split: str | None = "train",
        use_task_stage_mask: bool = True,
    ):
        self.stage_vocab = dict(stage_vocab)
        if self.stage_vocab.get(DONE_KEY) != DONE_ID or sorted(self.stage_vocab.values()) != list(
            range(len(stage_vocab))
        ):
            raise ValueError("Stage IDs must be contiguous with <DONE> at zero")
        self.num_stage_classes = len(stage_vocab)
        self.use_task_stage_mask = use_task_stage_mask
        self._episodes = {
            key: dict(value) for key, value in episode_metadata.items() if split is None or value["split"] == split
        }
        if not self._episodes:
            raise ValueError(f"No episodes found for split {split!r}")
        self._rows: dict[str, list[dict]] = {key: [] for key in self._episodes}
        self._starts = {}
        self._task_masks = {}
        for episode in self._episodes.values():
            mask = np.zeros(self.num_stage_classes, dtype=np.bool_)
            for stage_id in task_stage_vocab.get(episode["task_name"], []):
                if not 0 < stage_id < self.num_stage_classes:
                    raise ValueError("Task legal stage IDs must be real vocabulary stages")
                mask[stage_id] = True
            self._task_masks[episode["task_id"]] = mask
        for row in segments:
            key = _episode_key(row["episode_id"], row.get("dataset_index", 0))
            if split is not None and row["split"] != split:
                continue
            if key not in self._episodes:
                raise ValueError(f"Segment has no episode metadata: {key}")
            episode = self._episodes[key]
            if row["task_id"] != episode["task_id"] or row["task_name"] != episode["task_name"]:
                raise ValueError("Segment and episode task metadata differ")
            start = _integer(row["start_frame"], "start_frame")
            end = _integer(row["end_frame"], "end_frame")
            if not start < end <= episode["source_num_frames"]:
                raise ValueError(f"Invalid segment bounds in {key}")
            progress_start = _integer(row.get("progress_start_frame", start), "progress_start_frame")
            progress_end = _integer(row.get("progress_end_frame", end), "progress_end_frame")
            if not 0 <= progress_start <= start < end <= progress_end <= episode["source_num_frames"]:
                raise ValueError(f"Invalid original progress bounds in {key}")
            for id_key in ("current_stage_id", "next_stage_id"):
                stage_id = row[id_key]
                if stage_id != IGNORE_ID and not 0 <= stage_id < self.num_stage_classes:
                    raise ValueError(f"Invalid {id_key}: {stage_id}")
            if row["current_stage_id"] == DONE_ID:
                raise ValueError("<DONE> cannot be a current stage")
            self._rows[key].append(dict(row))
        for key, rows in self._rows.items():
            rows.sort(key=lambda row: row["start_frame"])
            for previous, current in pairwise(rows):
                if previous["end_frame"] > current["start_frame"]:
                    raise ValueError(f"Overlapping segments in {key}")
            self._starts[key] = [row["start_frame"] for row in rows]

    @classmethod
    def from_assets(
        cls, stage_assets_dir: str | Path, *, split: str | None = "train", use_task_stage_mask: bool = True
    ) -> StageAnnotationLookup:
        import polars as pl  # noqa: PLC0415

        directory = Path(stage_assets_dir)
        config = _read_json(directory / "stage_config.json")
        if (
            config.get("schema_version") != SCHEMA_VERSION
            or config.get("coordinate_system") != "source_frames"
            or config.get("interval_convention") != "half_open"
        ):
            raise ValueError("Unsupported OUC stage asset schema or frame coordinate system")
        vocabulary = _read_json(directory / "stage_vocab.json")
        if (
            config["num_stage_classes"] != len(vocabulary)
            or config["done_stage_id"] != DONE_ID
            or config["num_stage_queries"] != 3
        ):
            raise ValueError("stage_config.json disagrees with the vocabulary/query definition")
        return cls(
            pl.read_parquet(directory / "stage_segments.parquet").to_dicts(),
            _read_json(directory / "episode_stage_metadata.json"),
            vocabulary,
            _read_json(directory / "task_stage_vocab.json"),
            split=split,
            use_task_stage_mask=use_task_stage_mask,
        )

    def lookup(
        self,
        episode_id: Any,
        frame_idx: int | None = None,
        *,
        timestamp: float | None = None,
        source_frame_index: float | None = None,
        dataset_index: int = 0,
    ) -> dict[str, Any]:
        key = _episode_key(episode_id, dataset_index)
        if key not in self._episodes:
            raise KeyError(f"Episode {key} is absent from the requested stage split; check the manifest identity")
        episode = self._episodes[key]
        if frame_idx is not None:
            frame_idx = _integer(frame_idx, "frame_idx")
            if frame_idx >= episode["dataset_num_frames"]:
                raise ValueError(f"Dataset anchor {frame_idx} exceeds manifest bounds for {key}")
        if timestamp is not None:
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("timestamp must be finite, nonnegative, and relative to the dataset episode")
            if frame_idx is not None and source_frame_index is None:
                # Float32 dataset timestamps may accumulate sub-millisecond error.
                tolerance = max(1e-4, abs(timestamp) * 2e-7)
                if abs(timestamp - frame_idx / episode["dataset_fps"]) > tolerance:
                    raise ValueError("Dataset timestamp/frame_index disagrees with manifest dataset_fps")
            timestamp_frame = (timestamp + episode["timestamp_offset"]) * episode["source_fps"]
        else:
            timestamp_frame = None
        if source_frame_index is not None:
            anchor = float(source_frame_index)
            if timestamp_frame is not None and not math.isclose(anchor, timestamp_frame, abs_tol=1e-3, rel_tol=2e-7):
                raise ValueError("source_frame_index and timestamp map to different source frames")
        elif timestamp_frame is not None:
            anchor = timestamp_frame
        elif (
            frame_idx is not None
            and episode["source_fps"] == episode["dataset_fps"]
            and episode["timestamp_offset"] == 0
        ):
            anchor = float(frame_idx)
        else:
            raise ValueError("Unequal/offset frame clocks require a timestamp or explicit source_frame_index")
        # Recover exact integer boundaries from floating point timestamp arithmetic.
        if math.isfinite(anchor) and math.isclose(anchor, round(anchor), abs_tol=1e-3, rel_tol=0):
            anchor = float(round(anchor))
        if not math.isfinite(anchor) or not 0 <= anchor < episode["source_num_frames"]:
            raise ValueError(f"Source anchor {anchor} is outside episode {key}")
        targets = {
            "current_stage_id": np.int64(IGNORE_ID),
            "next_stage_id": np.int64(IGNORE_ID),
            "progress": np.float32(0),
            "stage_valid": np.bool_(0),
            "current_stage_valid": np.bool_(0),
            "next_stage_valid": np.bool_(0),
            "task_id": np.int64(episode["task_id"]),
        }
        if self.use_task_stage_mask:
            targets["allowed_stage_mask"] = self._task_masks[episode["task_id"]].copy()
        index = bisect_right(self._starts[key], anchor) - 1
        if index >= 0:
            row = self._rows[key][index]
            current, following = row["current_stage_id"], row["next_stage_id"]
            legal_mask = self._task_masks[episode["task_id"]]
            current_valid = current > 0 and (
                not self.use_task_stage_mask or legal_mask[current]
            ) and not row.get("is_ambiguous", False)
            next_valid = current_valid and following != IGNORE_ID and (
                not self.use_task_stage_mask
                or following == DONE_ID
                or (following > 0 and legal_mask[following])
            )
            if anchor < row["end_frame"] and current_valid:
                progress_start = row.get("progress_start_frame", row["start_frame"])
                progress_end = row.get("progress_end_frame", row["end_frame"])
                targets.update(
                    current_stage_id=np.int64(current),
                    progress=np.float32(np.clip(
                        (anchor - progress_start) / (progress_end - progress_start), 0, 1
                    )),
                    current_stage_valid=np.bool_(1),
                )
                if next_valid:
                    targets.update(
                        next_stage_id=np.int64(following),
                        next_stage_valid=np.bool_(1),
                        stage_valid=np.bool_(1),
                    )
        return targets


# Equivalent name for callers that prefer the segment-index terminology.
StageAnnotationIndex = StageAnnotationLookup
