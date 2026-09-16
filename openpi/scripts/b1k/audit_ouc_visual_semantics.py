#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import inspect
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/data2/yuxi.wang/yuxi_wang/xieshuaiyang")
DEFAULT_DATA = ROOT / "2026-challenge-demos"
DEFAULT_ASSETS = ROOT / "stage_assets_ouc"
DEFAULT_OUT = ROOT / "openpi/outputs/ouc_visual_audit"

REPO_ID = "behavior-1k/2026-challenge-demos"
DEFAULT_RGB_CAMERAS = [
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual semantic/boundary audit for OUC stage annotations."
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--repo-id", default=REPO_ID)

    # 语义覆盖：每个 185 类取一个最长、非冲突样本。
    p.add_argument("--class-coverage", action=argparse.BooleanOptionalAction, default=True)

    # 随机 episode 覆盖：每任务 5 个 episode，每 episode 抽 4 个 stage。
    p.add_argument("--episodes-per-task", type=int, default=5)
    p.add_argument("--segments-per-episode", type=int, default=4)

    # 边界专项：每任务随机抽 6 个连续 stage transition。
    p.add_argument("--transitions-per-task", type=int, default=6)

    p.add_argument("--seed", type=int, default=2026)

    # auto 优先 ZED/head/front 第三视角。
    # 也可以显式：
    # --camera zed
    # --camera observation.images.xxx
    p.add_argument(
        "--cameras",
        nargs="+",
        default=DEFAULT_RGB_CAMERAS,
        help="RGB cameras used for visual audit.",
    )
    # Contact sheet 中每个 frame 的显示宽度。
    p.add_argument("--frame-width", type=int, default=360)

    return p.parse_args()


def scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def episode_key(dataset_index, episode_id):
    return json.dumps(
        [int(dataset_index), str(episode_id)],
        separators=(",", ":"),
    )


def flatten_dict(obj, prefix=""):
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                result.update(flatten_dict(v, name))
            else:
                result[name] = v
    else:
        result[prefix] = obj
    return result


def value_is_image(v):
    if isinstance(v, Image.Image):
        return True

    try:
        shape = tuple(v.shape)
    except Exception:
        return False

    if len(shape) != 3:
        return False

    return (
        shape[-1] in (1, 3, 4)
        or shape[0] in (1, 3, 4)
    )


def choose_camera(sample, requested):
    flat = flatten_dict(sample)

    candidates = [
        k for k, v in flat.items()
        if "image" in k.lower() and value_is_image(v)
    ]

    if not candidates:
        # 有些版本视频 key 不一定带 image。
        candidates = [
            k for k, v in flat.items()
            if value_is_image(v)
        ]

    if not candidates:
        print("\nSample keys:")
        for k in sorted(flat):
            print(" ", k)
        raise RuntimeError(
            "No decoded image field found in LeRobot sample."
        )

    print("\nDetected image fields:")
    for k in candidates:
        print(" ", k)

    if requested != "auto":
        exact = [k for k in candidates if k == requested]
        if exact:
            return exact[0]

        partial = [
            k for k in candidates
            if requested.lower() in k.lower()
        ]
        if len(partial) == 1:
            return partial[0]

        if not partial:
            raise RuntimeError(
                f"--camera {requested!r} matched no image field."
            )

        raise RuntimeError(
            f"--camera {requested!r} matched multiple fields: {partial}"
        )

    def score(k):
        s = k.lower()
        value = 0

        # ------------------------------------------------
        # Modality: semantic audit MUST prefer RGB.
        # ------------------------------------------------
        if "rgb" in s:
            value += 1000

        if "depth" in s:
            value -= 1000

        # ------------------------------------------------
        # View: prefer third-person / main camera.
        # ------------------------------------------------
        if "zed" in s:
            value += 100

        if "head" in s:
            value += 90

        if "front" in s:
            value += 80

        if "main" in s:
            value += 70

        if "primary" in s:
            value += 60

        # Wrist / side cameras are lower priority.
        if "wrist" in s:
            value -= 60

        if "left_realsense" in s:
            value -= 20

        if "right_realsense" in s:
            value -= 20

        return value

        
    return max(candidates, key=score)


def to_pil(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(f"Expected 3D image, got {value.shape}")

    # CHW -> HWC
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.transpose(value, (1, 2, 0))

    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)

    if value.shape[-1] == 4:
        value = value[..., :3]

    if np.issubdtype(value.dtype, np.floating):
        finite = value[np.isfinite(value)]

        if finite.size == 0:
            value = np.zeros_like(value, dtype=np.uint8)

        elif finite.min() < -0.1:
            # [-1, 1]
            value = np.clip(
                (value + 1.0) * 127.5,
                0,
                255,
            ).astype(np.uint8)

        elif finite.max() <= 1.5:
            # [0, 1]
            value = np.clip(
                value * 255.0,
                0,
                255,
            ).astype(np.uint8)

        else:
            value = np.clip(
                value,
                0,
                255,
            ).astype(np.uint8)

    else:
        value = np.clip(
            value,
            0,
            255,
        ).astype(np.uint8)

    return Image.fromarray(value, mode="RGB")


def load_lerobot_dataset(repo_id, root, episode_ids):
    # 强制只用本地数据，避免审计脚本意外走网络。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sig = inspect.signature(LeRobotDataset)

    kwargs = {
        "repo_id": repo_id,
        "root": root,
    }

    if "tolerance_s" in sig.parameters:
        kwargs["tolerance_s"] = 5e-4

    if "episodes" in sig.parameters:
        kwargs["episodes"] = sorted(
            int(x) for x in episode_ids
        )

    if "video_backend" in sig.parameters:
        kwargs["video_backend"] = "pyav"

    print("\nLoading local LeRobot dataset:")
    print(" repo_id :", repo_id)
    print(" root    :", root)
    print(" episodes:", len(episode_ids))

    try:
        return LeRobotDataset(**kwargs)

    except TypeError as exc:
        # 个别 lerobot 版本没有 video_backend。
        kwargs.pop("video_backend", None)
        print(
            "Retry without video_backend due to:",
            exc,
        )
        return LeRobotDataset(**kwargs)


def get_hf_dataset(dataset):
    obj = dataset

    # 兼容可能存在的简单 wrapper
    for _ in range(10):
        if hasattr(obj, "hf_dataset"):
            return obj.hf_dataset
        if hasattr(obj, "_dataset"):
            obj = obj._dataset
            continue
        break

    raise RuntimeError(
        "Could not find hf_dataset inside LeRobotDataset."
    )


def create_sheet(
    cached_paths,
    frame_ids,
    camera_keys,
    title,
    subtitle,
    output_path,
    frame_width,
):
    """
    Rows    = cameras
    Columns = temporal frames
    """

    loaded = {}

    for camera_key in camera_keys:
        loaded[camera_key] = [
            Image.open(
                cached_paths[f][camera_key]
            ).convert("RGB")
            for f in frame_ids
        ]

    all_images = [
        image
        for images in loaded.values()
        for image in images
    ]

    max_ratio = max(
        img.height / img.width
        for img in all_images
    )

    frame_height = int(
        frame_width * max_ratio
    )

    top_h = 115
    row_label_h = 28
    bottom_label_h = 26

    row_height = (
        row_label_h
        + frame_height
        + bottom_label_h
    )

    canvas = Image.new(
        "RGB",
        (
            frame_width * len(frame_ids),
            top_h
            + row_height * len(camera_keys),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (10, 10),
        title[:220],
        fill="black",
        font=font,
    )

    wrapped = []
    line = ""

    for word in subtitle.split():
        candidate = (
            line + " " + word
        ).strip()

        if len(candidate) > 145:
            wrapped.append(line)
            line = word
        else:
            line = candidate

    if line:
        wrapped.append(line)

    for i, line in enumerate(
        wrapped[:3]
    ):
        draw.text(
            (10, 34 + i * 18),
            line,
            fill="black",
            font=font,
        )

    for row_idx, camera_key in enumerate(
        camera_keys
    ):
        base_y = (
            top_h
            + row_idx * row_height
        )

        short_name = (
            camera_key
            .replace(
                "observation.rgb.",
                "",
            )
            .replace(
                "_camera_0",
                "",
            )
        )

        draw.text(
            (8, base_y + 6),
            short_name,
            fill="black",
            font=font,
        )

        image_y = (
            base_y + row_label_h
        )

        for col, (
            img,
            frame,
        ) in enumerate(
            zip(
                loaded[camera_key],
                frame_ids,
            )
        ):
            thumb = ImageOps.contain(
                img,
                (
                    frame_width,
                    frame_height,
                ),
            )

            x = col * frame_width

            y = (
                image_y
                + (
                    frame_height
                    - thumb.height
                )
                // 2
            )

            canvas.paste(
                thumb,
                (
                    x
                    + (
                        frame_width
                        - thumb.width
                    )
                    // 2,
                    y,
                ),
            )

            draw.rectangle(
                (
                    x,
                    image_y,
                    x + frame_width - 1,
                    image_y
                    + frame_height - 1,
                ),
                outline="gray",
            )

            draw.text(
                (
                    x + 8,
                    image_y
                    + frame_height
                    + 7,
                ),
                f"frame={frame}",
                fill="black",
                font=font,
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        quality=92,
    )


def main():
    args = parse_args()

    rng = random.Random(args.seed)

    assets = args.assets
    data_root = args.data
    out = args.out

    out.mkdir(parents=True, exist_ok=True)

    sheet_dir = out / "sheets"
    cache_dir = out / "_frame_cache"

    sheet_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (assets / "manifest_ouc.json").read_text()
    )

    episodes = manifest["episodes"]

    manifest_by_ep = {
        str(e["episode_id"]): e
        for e in episodes
    }

    final = pl.read_parquet(
        assets / "stage_segments.parquet"
    ).to_dicts()

    raw = pl.read_parquet(
        assets / "raw_stage_segments.parquet"
    ).to_dicts()

    exceptional = json.loads(
        (
            assets
            / "stage_invalid_intervals.json"
        ).read_text()
    )

    vocab = json.loads(
        (assets / "stage_vocab.json").read_text()
    )

    id_to_stage = {
        int(v): k for k, v in vocab.items()
    }

    # ========================================================
    # Organize final rows
    # ========================================================

    valid_rows = [
        r for r in final
        if (
            not bool(r.get("is_ambiguous", False))
            and int(r["current_stage_id"]) > 0
        )
    ]

    by_episode = defaultdict(list)
    by_task = defaultdict(list)

    for r in final:
        by_episode[str(r["episode_id"])].append(r)
        by_task[r["task_name"]].append(r)

    for rows in by_episode.values():
        rows.sort(
            key=lambda r: int(r["start_frame"])
        )

    raw_by_ep_rid = {
        (
            str(r["episode_id"]),
            str(r["raw_segment_id"]),
        ): r
        for r in raw
    }

    items = []

    def add_item(
        kind,
        row,
        stage_text,
        stage_id,
        start,
        end,
        extra="",
        frame_mode="segment",
    ):
        eid = str(row["episode_id"])

        ep = manifest_by_ep[eid]
        n = int(ep["source_num_frames"])

        start = max(0, min(int(start), n - 1))
        end = max(start + 1, min(int(end), n))

        if frame_mode == "transition":
            boundary = end

            frames = [
                max(0, boundary - 30),
                max(0, boundary - 15),
                max(0, boundary - 5),
                max(0, boundary - 1),
                min(n - 1, boundary),
                min(n - 1, boundary + 1),
                min(n - 1, boundary + 5),
                min(n - 1, boundary + 15),
                min(n - 1, boundary + 30),
            ]

            frames = sorted(set(frames))

        else:
            frames = [
                max(0, start - 5),
                start,
                (start + end - 1) // 2,
                end - 1,
                min(n - 1, end + 5),
            ]

        uid = (
            f"{len(items):04d}_"
            f"{kind}_"
            f"ep{int(eid):06d}_"
            f"{start}_{end}"
        )

        items.append(
            {
                "uid": uid,
                "kind": kind,
                "task_id": int(row["task_id"]),
                "task_name": row["task_name"],
                "episode_id": eid,
                "stage_id": int(stage_id),
                "stage_text": str(stage_text),
                "start_frame": start,
                "end_frame": end,
                "duration": end - start,
                "frames": frames,
                "extra": extra,
            }
        )

    # ========================================================
    # A. At least one representative for every stage class
    # ========================================================

    if args.class_coverage:
        best = {}

        for r in valid_rows:
            sid = int(r["current_stage_id"])
            duration = (
                int(r["end_frame"])
                - int(r["start_frame"])
            )

            if (
                sid not in best
                or duration > best[sid][0]
            ):
                best[sid] = (duration, r)

        for sid in sorted(best):
            r = best[sid][1]

            add_item(
                "class",
                r,
                r.get(
                    "current_stage_text",
                    id_to_stage.get(sid, ""),
                ),
                sid,
                r["start_frame"],
                r["end_frame"],
                extra="Longest clean representative for this stage class.",
            )

    # ========================================================
    # B. Random episode-level coverage
    # ========================================================

    task_to_eps = defaultdict(list)

    for ep in episodes:
        task_to_eps[ep["task_name"]].append(
            str(ep["episode_id"])
        )

    for task_name in sorted(task_to_eps):
        eps = sorted(
            task_to_eps[task_name],
            key=int,
        )

        selected_eps = rng.sample(
            eps,
            min(args.episodes_per_task, len(eps)),
        )

        for eid in selected_eps:
            rows = [
                r for r in by_episode[eid]
                if (
                    not bool(
                        r.get(
                            "is_ambiguous",
                            False,
                        )
                    )
                    and int(
                        r["current_stage_id"]
                    ) > 0
                )
            ]

            if not rows:
                continue

            k = min(
                args.segments_per_episode,
                len(rows),
            )

            # 均匀覆盖整个 episode，不只抽开头。
            if k == 1:
                indices = [len(rows) // 2]
            else:
                indices = sorted(
                    {
                        round(
                            i
                            * (len(rows) - 1)
                            / (k - 1)
                        )
                        for i in range(k)
                    }
                )

            for idx in indices:
                r = rows[idx]

                add_item(
                    "episode",
                    r,
                    r.get(
                        "current_stage_text",
                        "",
                    ),
                    r["current_stage_id"],
                    r["start_frame"],
                    r["end_frame"],
                    extra=(
                        "Random episode coverage."
                    ),
                )

    # ========================================================
    # C. Transition boundary audit
    # ========================================================

    transitions_by_task = defaultdict(list)

    for eid, rows in by_episode.items():
        for a, b in zip(rows, rows[1:]):
            if (
                not bool(
                    a.get("is_ambiguous", False)
                )
                and not bool(
                    b.get("is_ambiguous", False)
                )
                and int(a["current_stage_id"]) > 0
                and int(b["current_stage_id"]) > 0
                and int(a["end_frame"])
                == int(b["start_frame"])
            ):
                transitions_by_task[
                    a["task_name"]
                ].append((a, b))

    for task_name in sorted(
        transitions_by_task
    ):
        candidates = transitions_by_task[
            task_name
        ]

        chosen = rng.sample(
            candidates,
            min(
                args.transitions_per_task,
                len(candidates),
            ),
        )

        for a, b in chosen:
            boundary = int(a["end_frame"])

            add_item(
                "transition",
                a,
                (
                    f"{a.get('current_stage_text','')}"
                    f"  ->  "
                    f"{b.get('current_stage_text','')}"
                ),
                a["current_stage_id"],
                max(
                    int(a["start_frame"]),
                    boundary - 15,
                ),
                boundary,
                extra=(
                    f"next_stage_id="
                    f"{int(b['current_stage_id'])}; "
                    f"boundary={boundary}"
                ),
                frame_mode="transition",
            )

    # ========================================================
    # D. ALL 48 conflict intervals
    # ========================================================

    conflicts = [
        r for r in final
        if bool(r.get("is_ambiguous", False))
    ]

    for r in conflicts:
        source_texts = []

        for rid in r.get(
            "raw_segment_ids",
            [],
        ):
            source = raw_by_ep_rid.get(
                (
                    str(r["episode_id"]),
                    str(rid),
                )
            )

            if source is not None:
                source_texts.append(
                    source.get(
                        "current_stage_text",
                        "",
                    )
                )

        add_item(
            "conflict",
            r,
            "AMBIGUOUS / CONFLICT",
            -100,
            r["start_frame"],
            r["end_frame"],
            extra=(
                "overlapping source stages: "
                + " || ".join(source_texts)
            ),
        )

    # ========================================================
    # E. ALL exceptional raw intervals: 1 reversed + 8 clipped
    # ========================================================

    for r in exceptional:
        extra = (
            f"is_invalid={r.get('is_invalid')}; "
            f"reason={r.get('invalid_reason')}; "
            f"adjustment={r.get('source_interval_adjustment')}; "
            f"raw=[{r.get('raw_start_frame')},"
            f"{r.get('raw_end_frame')})"
        )

        add_item(
            "exceptional",
            r,
            r.get("current_stage_text", ""),
            -100,
            r["start_frame"],
            r["end_frame"],
            extra=extra,
        )

    # ========================================================
    # Summary before video decode
    # ========================================================

    kinds = defaultdict(int)
    for item in items:
        kinds[item["kind"]] += 1

    print("\n======================================")
    print("VISUAL AUDIT SET")
    print("======================================")

    for kind, count in sorted(kinds.items()):
        print(f"{kind:12s}: {count}")

    print("total       :", len(items))

    all_pairs = sorted(
        {
            (
                int(item["episode_id"]),
                int(frame),
            )
            for item in items
            for frame in item["frames"]
        }
    )

    print(
        "unique RGB frames to decode:",
        len(all_pairs),
    )

    # ========================================================
    # Load exact local LeRobot videos
    # ========================================================

    episode_ids = {
        int(e["episode_id"])
        for e in episodes
    }

    dataset = load_lerobot_dataset(
        args.repo_id,
        data_root,
        episode_ids,
    )

    hf = get_hf_dataset(dataset)

    # Build:
    # (episode_index, frame_index) -> local dataset row
    #
    # This avoids assuming global LeRobot `index`
    # equals local dataset __getitem__ index.
    arrow = hf.data.table.select(
        [
            "episode_index",
            "frame_index",
        ]
    )

    identity_df = (
        pl.from_arrow(arrow)
        .with_row_index("_local_index")
    )

    target_df = pl.DataFrame(
        {
            "episode_index": [
                e for e, _ in all_pairs
            ],
            "frame_index": [
                f for _, f in all_pairs
            ],
        }
    )

    matched = target_df.join(
        identity_df,
        on=[
            "episode_index",
            "frame_index",
        ],
        how="left",
    )

    missing = matched.filter(
        pl.col("_local_index").is_null()
    )

    if missing.height:
        print(missing.head(30))
        raise RuntimeError(
            f"{missing.height} requested visual frames "
            "are absent from LeRobotDataset."
        )

    local_index = {
        (
            int(r["episode_index"]),
            int(r["frame_index"]),
        ): int(r["_local_index"])
        for r in matched.iter_rows(named=True)
    }

    # ========================================================
    # Detect actual camera from a REAL decoded sample
    # ========================================================

    first_pair = all_pairs[0]

    first_sample = dataset[
        local_index[first_pair]
    ]

    flat_first = flatten_dict(first_sample)

    camera_keys = list(args.cameras)

    for camera_key in camera_keys:
        if camera_key not in flat_first:
            raise KeyError(
                f"Requested camera {camera_key!r} "
                "is absent from decoded sample."
            )

        if not value_is_image(flat_first[camera_key]):
            raise TypeError(
                f"{camera_key!r} is not an image field."
            )

    print("\nSelected RGB cameras:")
    for camera_key in camera_keys:
        print(" ", camera_key)

    # ========================================================
    # Decode only required unique frames and cache thumbnails
    # ========================================================

    pair_to_cache = {}

    for i, pair in enumerate(all_pairs, start=1):
        eid, frame = pair

        pair_to_cache[pair] = {}

        expected_paths = {}

        for camera_key in camera_keys:
            short_name = (
                camera_key
                .replace("observation.rgb.", "")
                .replace("_camera_0", "")
            )

            cache_path = (
                cache_dir
                / short_name
                / f"ep{eid:06d}_f{frame:06d}.jpg"
            )

            expected_paths[camera_key] = cache_path
            pair_to_cache[pair][camera_key] = cache_path

        # 三个都已经缓存，则不用重新读 dataset
        if all(
            p.is_file()
            for p in expected_paths.values()
        ):
            continue

        sample = dataset[
            local_index[pair]
        ]

        flat = flatten_dict(sample)

        for camera_key in camera_keys:

            if camera_key not in flat:
                raise KeyError(
                    f"Camera {camera_key!r} missing "
                    f"from episode={eid}, frame={frame}"
                )

            cache_path = expected_paths[camera_key]

            if cache_path.is_file():
                continue

            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            image = to_pil(
                flat[camera_key]
            )

            max_w = 720

            if image.width > max_w:
                h = round(
                    image.height
                    * max_w
                    / image.width
                )

                image = image.resize(
                    (max_w, h),
                    Image.Resampling.LANCZOS,
                )

            image.save(
                cache_path,
                quality=90,
            )

        if i % 100 == 0 or i == len(all_pairs):
            print(
                f"decoded {i}/{len(all_pairs)} frame timestamps "
                f"x {len(camera_keys)} RGB views"
            )

            # 保存一个审计分辨率版本，避免数千张原图占太多空间
            max_w = 720

            if image.width > max_w:
                h = round(
                    image.height
                    * max_w
                    / image.width
                )
                image = image.resize(
                    (max_w, h),
                    Image.Resampling.LANCZOS,
                )

            image.save(
                cache_path,
                quality=90,
            )

            if i % 100 == 0 or i == len(all_pairs):
                print(
                    f"decoded {i}/{len(all_pairs)} frames"
                )

    # ========================================================
    # Build contact sheets
    # ========================================================

    for i, item in enumerate(items, start=1):
        eid = int(item["episode_id"])

        cache_paths = {
            f: pair_to_cache[(eid, f)]
            for f in item["frames"]
        }

        sheet_path = (
            sheet_dir
            / f"{item['uid']}.jpg"
        )

        title = (
            f"[{item['kind']}] "
            f"task={item['task_name']} | "
            f"episode={eid} | "
            f"stage_id={item['stage_id']} | "
            f"[{item['start_frame']},{item['end_frame']})"
        )

        subtitle = (
            f"{item['stage_text']} | "
            f"{item['extra']}"
        )

        create_sheet(
            cache_paths,
            item["frames"],
            camera_keys,
            title,
            subtitle,
            sheet_path,
            args.frame_width,
        )

        item["sheet"] = (
            Path("sheets")
            / sheet_path.name
        ).as_posix()

        if i % 100 == 0 or i == len(items):
            print(
                f"built {i}/{len(items)} contact sheets"
            )

    # ========================================================
    # Write machine-readable audit manifest
    # ========================================================

    csv_path = out / "audit_items.csv"

    fields = [
        "uid",
        "kind",
        "task_id",
        "task_name",
        "episode_id",
        "stage_id",
        "stage_text",
        "start_frame",
        "end_frame",
        "duration",
        "frames",
        "extra",
        "sheet",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for item in items:
            row = dict(item)
            row["frames"] = json.dumps(
                row["frames"]
            )
            writer.writerow(
                {
                    k: row[k]
                    for k in fields
                }
            )

    # ========================================================
    # Interactive HTML reviewer
    # ========================================================

    cards = []

    for item in items:
        uid = html.escape(item["uid"])
        stage = html.escape(
            item["stage_text"]
        )
        extra = html.escape(
            item["extra"]
        )

        cards.append(
            f"""
<div class="card" data-uid="{uid}">
  <h3>
    {html.escape(item["kind"])}
    · {html.escape(item["task_name"])}
    · ep {item["episode_id"]}
    · stage {item["stage_id"]}
    · [{item["start_frame"]},{item["end_frame"]})
  </h3>

  <div class="stage">{stage}</div>
  <div class="extra">{extra}</div>

  <img src="{html.escape(item["sheet"])}">

  <div class="review">
    <label>
      Semantic:
      <select data-field="semantic">
        <option value=""></option>
        <option>PASS</option>
        <option>FAIL</option>
        <option>UNCLEAR</option>
      </select>
    </label>

    <label>
      Start boundary:
      <select data-field="start_boundary">
        <option value=""></option>
        <option>PASS</option>
        <option>EARLY</option>
        <option>LATE</option>
        <option>UNCLEAR</option>
        <option>N/A</option>
      </select>
    </label>

    <label>
      End boundary:
      <select data-field="end_boundary">
        <option value=""></option>
        <option>PASS</option>
        <option>EARLY</option>
        <option>LATE</option>
        <option>UNCLEAR</option>
        <option>N/A</option>
      </select>
    </label>

    <label>
      Overall:
      <select data-field="overall">
        <option value=""></option>
        <option>PASS</option>
        <option>FAIL</option>
        <option>UNCLEAR</option>
      </select>
    </label>

    <input
      data-field="notes"
      placeholder="notes..."
      style="width:420px"
    >
  </div>
</div>
"""
        )

    items_json = json.dumps(
        items,
        ensure_ascii=False,
    )

    counts_html = " · ".join(
        f"{html.escape(k)}={v}"
        for k, v in sorted(kinds.items())
    )

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OUC Visual Semantic Audit</title>

<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  background: #f5f5f5;
}}

.header {{
  position: sticky;
  top: 0;
  background: white;
  padding: 14px;
  border: 1px solid #ccc;
  z-index: 10;
}}

.card {{
  background: white;
  margin: 18px 0;
  padding: 14px;
  border: 1px solid #ccc;
}}

.card img {{
  width: 100%;
  max-width: 1900px;
  display: block;
  margin: 12px 0;
}}

.stage {{
  font-weight: bold;
  margin-top: 5px;
}}

.extra {{
  margin-top: 5px;
  color: #555;
}}

.review {{
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  align-items: center;
}}

select, input, button {{
  padding: 5px;
}}

.stats {{
  margin-top: 8px;
}}
</style>
</head>

<body>

<div class="header">
  <b>OUC Visual Semantic Annotation Audit</b><br>
  cameras: {html.escape(" | ".join(camera_keys))}<br>
  total items: {len(items)} · {counts_html}

  <div class="stats" id="stats"></div>

  <button onclick="exportCSV()">
    Export review CSV
  </button>

  <button onclick="showOnly('all')">
    All
  </button>

  <button onclick="showOnly('unreviewed')">
    Unreviewed
  </button>

  <button onclick="showOnly('fail')">
    FAIL only
  </button>
</div>

{''.join(cards)}

<script>
const ITEMS = {items_json};
const KEY = "ouc_visual_stage_audit_v1";

let saved = JSON.parse(
  localStorage.getItem(KEY) || "{{}}"
);

function store() {{
  localStorage.setItem(
    KEY,
    JSON.stringify(saved)
  );
  refreshStats();
}}

document.querySelectorAll(".card").forEach(card => {{
  const uid = card.dataset.uid;

  if (!saved[uid]) saved[uid] = {{}};

  card.querySelectorAll("[data-field]").forEach(el => {{
    const field = el.dataset.field;

    if (saved[uid][field] !== undefined) {{
      el.value = saved[uid][field];
    }}

    el.addEventListener("change", () => {{
      saved[uid][field] = el.value;
      store();
    }});

    el.addEventListener("input", () => {{
      saved[uid][field] = el.value;
      store();
    }});
  }});
}});

function refreshStats() {{
  let reviewed = 0;
  let pass = 0;
  let fail = 0;
  let unclear = 0;

  ITEMS.forEach(item => {{
    const r = saved[item.uid] || {{}};

    if (r.overall) reviewed++;

    if (r.overall === "PASS") pass++;
    if (r.overall === "FAIL") fail++;
    if (r.overall === "UNCLEAR") unclear++;
  }});

  document.getElementById("stats").innerText =
    `reviewed=${{reviewed}}/${{ITEMS.length}} · ` +
    `PASS=${{pass}} · FAIL=${{fail}} · UNCLEAR=${{unclear}}`;
}}

function csvEscape(x) {{
  x = String(x ?? "");
  return '"' + x.replaceAll('"', '""') + '"';
}}

function exportCSV() {{
  const fields = [
    "uid",
    "kind",
    "task_id",
    "task_name",
    "episode_id",
    "stage_id",
    "stage_text",
    "start_frame",
    "end_frame",
    "semantic",
    "start_boundary",
    "end_boundary",
    "overall",
    "notes"
  ];

  let lines = [fields.join(",")];

  ITEMS.forEach(item => {{
    const r = saved[item.uid] || {{}};

    const row = {{
      ...item,
      semantic: r.semantic || "",
      start_boundary: r.start_boundary || "",
      end_boundary: r.end_boundary || "",
      overall: r.overall || "",
      notes: r.notes || ""
    }};

    lines.push(
      fields.map(f => csvEscape(row[f])).join(",")
    );
  }});

  const blob = new Blob(
    [lines.join("\\n")],
    {{type: "text/csv;charset=utf-8"}}
  );

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ouc_visual_review.csv";
  a.click();

  URL.revokeObjectURL(a.href);
}}

function showOnly(mode) {{
  document.querySelectorAll(".card").forEach(card => {{
    const uid = card.dataset.uid;
    const r = saved[uid] || {{}};

    let show = true;

    if (mode === "unreviewed")
      show = !r.overall;

    if (mode === "fail")
      show = r.overall === "FAIL";

    card.style.display = show ? "" : "none";
  }});
}}

refreshStats();
store();
</script>

</body>
</html>
"""

    html_path = out / "index.html"
    html_path.write_text(
        page,
        encoding="utf-8",
    )

    summary = {
        "cameras": camera_keys,
        "num_items": len(items),
        "num_unique_frames": len(all_pairs),
        "by_kind": dict(kinds),
        "output": str(out),
        "html": str(html_path),
    }

    (
        out / "audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("======================================")
    print("VISUAL AUDIT EXPORT COMPLETE")
    print("======================================")
    print("cameras:")
    for camera_key in camera_keys:
        print("  ", camera_key)
    print("audit items   :", len(items))
    print("unique frames :", len(all_pairs))
    print("HTML          :", html_path)
    print("CSV           :", csv_path)
    print("sheets        :", sheet_dir)
    print()
    print("Open index.html in a browser and review:")
    print("  Semantic")
    print("  Start boundary")
    print("  End boundary")
    print("  Overall")
    print()
    print("RESULT: READY FOR HUMAN AUDIT")


if __name__ == "__main__":
    main()
