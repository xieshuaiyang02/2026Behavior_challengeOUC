#!/usr/bin/env python3
"""Build custom OUC stage assets from an existing, verified manifest.

This is a replacement entry point, not the original lost script.
It preserves global episode IDs and the manifest's explicit annotation paths.
No Hugging Face download, episode renumbering, or annotation mutation occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow direct execution from the OpenPI repository root or another cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from openpi.training import stage_annotations_ouc as ouc


DEFAULT_ROOT = Path(
    "/data2/yuxi.wang/yuxi_wang/xieshuaiyang/2026-challenge-demos"
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_manifest_path(value, manifest_dir: Path) -> str:
    """Resolve an explicitly recorded path; never infer an annotation name."""
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def prepare_manifest(
    manifest_path: Path,
    dataset_root: Path,
    task_ids: set[int],
) -> dict:
    manifest_path = manifest_path.resolve()
    dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Provide --manifest with the original verified manifest."
        )

    manifest = read_json(manifest_path)
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest must contain a nonempty episodes list")

    selected = []
    seen = set()
    found_tasks = set()

    for original in episodes:
        task_id = int(original["task_id"])
        if task_id not in task_ids:
            continue

        ep = dict(original)
        episode_id = str(ep["episode_id"])
        dataset_index = int(ep.get("dataset_index", 0))
        identity = (dataset_index, episode_id)

        if identity in seen:
            raise ValueError(f"Duplicate episode identity: {identity}")
        seen.add(identity)

        # Keep the original global episode identity.
        ep["episode_id"] = episode_id
        ep["dataset_index"] = dataset_index
        ep["task_id"] = task_id

        # The manifest is the authority for the annotation mapping.
        if "annotation" not in ep:
            if "annotation_path" not in ep:
                raise ValueError(
                    f"Episode {identity} has no annotation or annotation_path"
                )
            path = Path(
                resolve_manifest_path(
                    ep["annotation_path"], manifest_path.parent
                )
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Annotation missing for {identity}: {path}"
                )
            ep["annotation_path"] = str(path)

        if "object_metadata_path" in ep:
            path = Path(
                resolve_manifest_path(
                    ep["object_metadata_path"], manifest_path.parent
                )
            )
            if not path.is_file():
                raise FileNotFoundError(f"Object metadata missing: {path}")
            ep["object_metadata_path"] = str(path)

        selected.append(ep)
        found_tasks.add(task_id)

    missing = task_ids - found_tasks
    if missing:
        raise ValueError(f"Requested task IDs absent from manifest: {sorted(missing)}")
    if not selected:
        raise ValueError("No episodes selected")

    return {
        "episodes": selected,
        "source_manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Existing verified manifest_ouc.json.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and build in a temporary directory without writing the destination.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else root.parent / "stage_assets_ouc" / "manifest_ouc.json"
    )

    manifest = prepare_manifest(
        manifest_path,
        root,
        set(args.task_ids),
    )

    print("OUC schema:", ouc.SCHEMA_VERSION)
    print("Manifest:", manifest_path)
    print("Selected episodes:", len(manifest["episodes"]))
    print("Task IDs:", sorted(set(args.task_ids)))

    if ouc.SCHEMA_VERSION != 2:
        raise RuntimeError("This builder expects OUC schema 2")

    # Use a temporary manifest with absolute annotation paths.
    # This also makes the saved manifest portable between output directories
    # on the same server without changing episode identities.
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.check_only and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}\n"
            "Choose a new output directory; existing assets will not be overwritten."
        )

    with tempfile.TemporaryDirectory(
        prefix="ouc_build_",
        dir=output.parent,
    ) as temp:
        temp = Path(temp)
        temp_manifest = temp / "manifest_ouc.json"
        write_json(temp_manifest, manifest)

        build_output = temp / "check_assets" if args.check_only else output

        result = ouc.build_stage_assets(temp_manifest, build_output)

        if args.check_only:
            print("Preflight result:", result)
            print("Preflight OK; destination unchanged.")
            return

        # Keep the exact manifest used for this build.
        write_json(output / "manifest_ouc.json", manifest)

    print("Build result:", result)
    print("Assets:", output)


if __name__ == "__main__":
    main()