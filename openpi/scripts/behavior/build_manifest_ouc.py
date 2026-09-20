#!/usr/bin/env python3
"""Build manifest_ouc.json from official BEHAVIOR 2026 episode metadata.

This script ONLY builds the manifest.

It:
1. Reads official meta/info.json.
2. Reads meta/episodes/**/*.parquet (or episodes.jsonl as fallback).
3. Selects episodes belonging to requested task IDs.
4. Preserves the official global episode_index as episode_id.
5. Uses the official annotation_path from metadata.
6. Verifies every annotation JSON exists locally.
7. Writes a manifest compatible with stage_annotations_ouc.py.

It does NOT:
- download anything;
- renumber episodes;
- infer annotation filenames;
- modify annotation JSONs;
- build stage assets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path(
    "/data2/yuxi.wang/yuxi_wang/xieshuaiyang/2026-challenge-demos"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_episode_metadata(dataset_root: Path) -> list[dict]:
    """Read official episode metadata.

    Preferred layout:
        meta/episodes/**/*.parquet

    Fallback:
        meta/episodes.jsonl
    """

    meta_root = dataset_root / "meta"
    parquet_root = meta_root / "episodes"

    parquet_files = sorted(parquet_root.glob("**/*.parquet"))

    if parquet_files:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required to read episode parquet files.\n"
                "Install with: pip install pyarrow"
            ) from exc

        rows: list[dict] = []

        wanted_columns = {
            "episode_index",
            "tasks",
            "task_index",
            "length",
            "annotation_path",
        }

        for path in parquet_files:
            schema = pq.read_schema(path)

            available = [
                key
                for key in wanted_columns
                if key in schema.names
            ]

            required = {
                "episode_index",
                "tasks",
                "task_index",
                "length",
                "annotation_path",
            }

            missing = required - set(available)

            if missing:
                raise ValueError(
                    f"{path} is missing required columns: "
                    f"{sorted(missing)}"
                )

            table = pq.read_table(
                path,
                columns=available,
            )

            rows.extend(table.to_pylist())

        if not rows:
            raise ValueError(
                f"No episode records found under {parquet_root}"
            )

        return rows

    # Fallback for alternative dataset layouts.
    jsonl_path = meta_root / "episodes.jsonl"

    if jsonl_path.is_file():
        rows = []

        for line in jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if line:
                rows.append(json.loads(line))

        if not rows:
            raise ValueError(
                f"No episode records found in {jsonl_path}"
            )

        return rows

    raise FileNotFoundError(
        "Could not find episode metadata.\n"
        f"Expected either:\n"
        f"  {parquet_root}/**/*.parquet\n"
        f"or:\n"
        f"  {jsonl_path}"
    )


def resolve_annotation_path(
    dataset_root: Path,
    annotation_path: str,
    annotation_root: Path | None = None,
) -> Path:
    """Resolve the official annotation_path.

    Important:
    annotation_path comes directly from official metadata.

    Example:
        annotations/task-0000/episode_00000010.json

    We NEVER infer the annotation filename from episode_index.
    """

    path = Path(annotation_path)

    if path.is_absolute():
        return path.resolve()

    if annotation_root is not None:
        # annotation_root should directly contain:
        #
        # task-0000/
        # task-0004/
        # ...
        #
        parts = path.parts

        if parts and parts[0] == "annotations":
            parts = parts[1:]

        return annotation_root.joinpath(*parts).resolve()

    return (dataset_root / path).resolve()


def validate_task_record(
    row: dict,
) -> tuple[int, str, int, int, str]:
    """Validate one official metadata record."""

    required = (
        "episode_index",
        "task_index",
        "tasks",
        "length",
        "annotation_path",
    )

    for key in required:
        if key not in row:
            raise ValueError(
                f"Episode metadata missing {key}: {row}"
            )

    episode_index = int(row["episode_index"])
    task_id = int(row["task_index"])
    length = int(row["length"])

    if episode_index < 0:
        raise ValueError(
            f"Invalid episode_index: {episode_index}"
        )

    if task_id < 0:
        raise ValueError(
            f"Invalid task_index: {task_id}"
        )

    if length <= 0:
        raise ValueError(
            f"Invalid episode length: {length}"
        )

    tasks = row["tasks"]

    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError(
            f"Expected exactly one task for episode "
            f"{episode_index}, got {tasks!r}"
        )

    task_name = tasks[0]

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError(
            f"Invalid task name for episode {episode_index}: "
            f"{task_name!r}"
        )

    annotation_path = row["annotation_path"]

    if (
        not isinstance(annotation_path, str)
        or not annotation_path.strip()
    ):
        raise ValueError(
            f"Episode {episode_index} has no valid "
            "annotation_path"
        )

    return (
        episode_index,
        task_name,
        task_id,
        length,
        annotation_path,
    )


def build_manifest(
    dataset_root: Path,
    task_ids: set[int],
    *,
    annotation_root: Path | None = None,
) -> tuple[dict, dict]:
    """Build manifest for selected task IDs."""

    dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    info_path = dataset_root / "meta" / "info.json"

    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing official metadata: {info_path}"
        )

    info = read_json(info_path)

    if "fps" not in info:
        raise ValueError(
            f"{info_path} does not contain fps"
        )

    fps = float(info["fps"])

    if fps <= 0:
        raise ValueError(
            f"Invalid dataset fps: {fps}"
        )

    rows = read_episode_metadata(dataset_root)

    manifest_episodes: list[dict] = []

    seen_episode_ids: set[int] = set()
    found_task_ids: set[int] = set()

    task_names_by_id: dict[int, str] = {}

    missing_annotations: list[str] = []

    for row in rows:
        (
            episode_index,
            task_name,
            task_id,
            length,
            annotation_relpath,
        ) = validate_task_record(row)

        if task_id not in task_ids:
            continue

        if episode_index in seen_episode_ids:
            raise ValueError(
                f"Duplicate global episode_index: "
                f"{episode_index}"
            )

        seen_episode_ids.add(episode_index)
        found_task_ids.add(task_id)

        if task_id in task_names_by_id:
            if task_names_by_id[task_id] != task_name:
                raise ValueError(
                    f"Task ID {task_id} maps to "
                    f"multiple task names:\n"
                    f"  {task_names_by_id[task_id]}\n"
                    f"  {task_name}"
                )
        else:
            task_names_by_id[task_id] = task_name

        annotation_path = resolve_annotation_path(
            dataset_root,
            annotation_relpath,
            annotation_root=annotation_root,
        )

        if not annotation_path.is_file():
            missing_annotations.append(
                str(annotation_path)
            )
            continue

        # -------------------------------------------------
        # Manifest fields required by stage_annotations_ouc
        # -------------------------------------------------
        #
        # Official dataset is currently used directly at
        # its native frame rate, therefore source clock and
        # dataset clock are identical.
        #
        episode = {
            "episode_id": str(episode_index),

            # Keep explicit even though default is zero.
            "dataset_index": 0,

            "task_id": task_id,
            "task_name": task_name,

            "split": "train",

            "source_fps": fps,
            "dataset_fps": fps,

            "source_num_frames": length,
            "dataset_num_frames": length,

            "timestamp_offset": 0.0,

            # Use absolute path so manifest can live
            # outside the dataset directory safely.
            "annotation_path": str(
                annotation_path.resolve()
            ),
        }

        manifest_episodes.append(episode)

    missing_tasks = task_ids - found_task_ids

    if missing_tasks:
        raise ValueError(
            "Requested task IDs were not found in "
            f"meta/episodes: {sorted(missing_tasks)}"
        )

    if missing_annotations:
        preview = "\n".join(
            f"  {p}"
            for p in missing_annotations[:20]
        )

        extra = ""

        if len(missing_annotations) > 20:
            extra = (
                f"\n  ... plus "
                f"{len(missing_annotations) - 20} more"
            )

        raise FileNotFoundError(
            f"{len(missing_annotations)} selected episodes "
            "are missing annotation JSON files:\n"
            f"{preview}"
            f"{extra}\n\n"
            "Download the corresponding annotations "
            "before building the manifest."
        )

    if not manifest_episodes:
        raise ValueError(
            "No episodes selected."
        )

    # Stable ordering.
    manifest_episodes.sort(
        key=lambda ep: (
            ep["task_id"],
            int(ep["episode_id"]),
        )
    )

    manifest = {
        "schema": "ouc_manifest_v2",
        "dataset_root": str(dataset_root),
        "episodes": manifest_episodes,
    }

    counts = Counter(
        ep["task_id"]
        for ep in manifest_episodes
    )

    frame_counts = defaultdict(int)

    for ep in manifest_episodes:
        frame_counts[ep["task_id"]] += (
            ep["dataset_num_frames"]
        )

    audit = {
        "dataset_root": str(dataset_root),
        "dataset_fps": fps,

        "selected_task_ids": sorted(task_ids),

        "num_tasks": len(task_ids),
        "num_episodes": len(manifest_episodes),

        "task_episode_counts": {
            str(task_id): counts[task_id]
            for task_id in sorted(counts)
        },

        "task_frame_counts": {
            str(task_id): frame_counts[task_id]
            for task_id in sorted(frame_counts)
        },

        "task_names": {
            str(task_id): task_names_by_id[task_id]
            for task_id in sorted(task_names_by_id)
            if task_id in task_ids
        },
    }

    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Official BEHAVIOR 2026 dataset root."
        ),
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Official task_index values to include. "
            "Example: 0 4 14 30 35 38 42 64"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output path for manifest_ouc.json."
        ),
    )

    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help=(
            "Optional directory directly containing "
            "task-XXXX annotation folders. "
            "Normally unnecessary when annotations "
            "are under dataset_root/annotations/."
        ),
    )

    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for selection audit."
        ),
    )

    args = parser.parse_args()

    task_ids = set(args.task_ids)

    if not task_ids:
        raise ValueError(
            "--task-ids must not be empty"
        )

    if any(task_id < 0 for task_id in task_ids):
        raise ValueError(
            "Task IDs must be nonnegative"
        )

    annotation_root = (
        args.annotation_root.resolve()
        if args.annotation_root is not None
        else None
    )

    manifest, audit = build_manifest(
        args.dataset_root,
        task_ids,
        annotation_root=annotation_root,
    )

    output = args.output.resolve()

    write_json(
        output,
        manifest,
    )

    if args.audit_output is not None:
        audit_output = (
            args.audit_output.resolve()
        )

        write_json(
            audit_output,
            audit,
        )

    print()
    print("========================================")
    print("OUC manifest generated successfully")
    print("========================================")
    print(f"Dataset root : {audit['dataset_root']}")
    print(f"Manifest     : {output}")
    print(
        f"Task IDs     : "
        f"{audit['selected_task_ids']}"
    )
    print(
        f"Tasks        : "
        f"{audit['num_tasks']}"
    )
    print(
        f"Episodes     : "
        f"{audit['num_episodes']}"
    )
    print(
        f"FPS          : "
        f"{audit['dataset_fps']}"
    )

    print()

    for task_id in audit["selected_task_ids"]:
        key = str(task_id)

        print(
            f"task {task_id:04d}: "
            f"{audit['task_names'].get(key, '?')} | "
            f"{audit['task_episode_counts'].get(key, 0)} episodes | "
            f"{audit['task_frame_counts'].get(key, 0)} frames"
        )

    print()
    print("No stage assets were generated.")
    print(
        "Next step: pass this manifest to "
        "build_stage_assets()."
    )


if __name__ == "__main__":
    main()