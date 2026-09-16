from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import polars as pl


ROOT = Path("/data2/yuxi.wang/yuxi_wang/xieshuaiyang")
OPENPI = ROOT / "openpi"
ASSETS = ROOT / "stage_assets_ouc"

MANIFEST = ASSETS / "manifest_ouc.json"

# 如果 manifest 实际在其他目录，可以自动回退
if not MANIFEST.exists():
    MANIFEST = ROOT / "manifest_ouc.json"

sys.path.insert(0, str(OPENPI))

from openpi.training.stage_annotations_ouc import (
    DONE_ID,
    DONE_KEY,
    IGNORE_ID,
    build_stage_key,
    stage_text,
)
from openpi.training.stage_supervision_ouc import (
    StageSupervisionLookup,
    TRAINING_TARGET_POLICY,
)


def fail(msg):
    raise RuntimeError(msg)


def ep_key(dataset_index, episode_id):
    return json.dumps(
        [int(dataset_index), str(episode_id)],
        separators=(",", ":"),
    )


def parse_annotation_intervals(duration):
    """
    Independent parser for the official frame_duration representation.

    Supports:
      [start, end]
    or
      [[start1, end1], [start2, end2], ...]
    """
    if (
        isinstance(duration, (list, tuple))
        and len(duration) == 2
        and not any(isinstance(x, (list, tuple)) for x in duration)
    ):
        return [duration]

    if (
        isinstance(duration, (list, tuple))
        and len(duration) > 0
        and all(
            isinstance(x, (list, tuple)) and len(x) == 2
            for x in duration
        )
    ):
        return list(duration)

    raise ValueError(f"Invalid frame_duration: {duration!r}")


# ============================================================
# Load assets
# ============================================================

manifest = json.loads(MANIFEST.read_text())
episodes = manifest["episodes"]

raw = pl.read_parquet(
    ASSETS / "raw_stage_segments.parquet"
).to_dicts()

final = pl.read_parquet(
    ASSETS / "stage_segments.parquet"
).to_dicts()

config = json.loads(
    (ASSETS / "stage_config.json").read_text()
)

vocab = json.loads(
    (ASSETS / "stage_vocab.json").read_text()
)

task_vocab = json.loads(
    (ASSETS / "task_stage_vocab.json").read_text()
)

conflicts_file = json.loads(
    (ASSETS / "stage_conflicts.json").read_text()
)

invalid_file = json.loads(
    (ASSETS / "stage_invalid_intervals.json").read_text()
)


print("==============================================")
print("OUC ASSET GENERATION CORRECTNESS AUDIT")
print("==============================================")
print("manifest episodes :", len(episodes))
print("raw segments      :", len(raw))
print("final segments    :", len(final))
print("stage classes     :", len(vocab))
print()


# ============================================================
# 1. Manifest sanity
# ============================================================

manifest_by_key = {}
task_counts = defaultdict(int)

for ep in episodes:
    key = ep_key(
        ep.get("dataset_index", 0),
        ep["episode_id"],
    )

    if key in manifest_by_key:
        fail(f"Duplicate manifest episode: {key}")

    manifest_by_key[key] = ep
    task_counts[
        (int(ep["task_id"]), ep["task_name"])
    ] += 1


print("----- Manifest tasks -----")

for task, n in sorted(task_counts.items()):
    print(task, n)

assert len(episodes) == 1600
assert len(task_counts) == 8
assert all(n == 200 for n in task_counts.values())

print("[PASS] manifest = 8 tasks x 200 episodes")
print()


# ============================================================
# 2. Original annotation -> raw_stage_segments
# ============================================================

raw_by_identity = {}

for row in raw:
    identity = (
        int(row.get("dataset_index", 0)),
        str(row["episode_id"]),
        int(row["skill_idx"]),
        int(row["interval_idx"]),
    )

    if identity in raw_by_identity:
        fail(f"Duplicate raw identity: {identity}")

    raw_by_identity[identity] = row


expected_raw_count = 0
raw_errors = []

expected_invalid = []
expected_clipped = []

annotation_stage_keys = set()


for ep in episodes:

    dataset_index = int(ep.get("dataset_index", 0))
    episode_id = str(ep["episode_id"])

    annotation_path = Path(ep["annotation_path"])

    if not annotation_path.is_file():
        raw_errors.append(
            (
                "missing_annotation_file",
                episode_id,
                str(annotation_path),
            )
        )
        continue

    annotation = json.loads(annotation_path.read_text())

    skills = annotation.get("skill_annotation")

    if not isinstance(skills, list):
        raw_errors.append(
            ("invalid_skill_annotation", episode_id)
        )
        continue

    seen_skill_idx = set()

    for default_idx, skill in enumerate(skills):

        skill_idx = int(
            skill.get("skill_idx", default_idx)
        )

        if skill_idx in seen_skill_idx:
            raw_errors.append(
                (
                    "duplicate_skill_idx",
                    episode_id,
                    skill_idx,
                )
            )
            continue

        seen_skill_idx.add(skill_idx)

        intervals = parse_annotation_intervals(
            skill.get("frame_duration")
        )

        # Canonical semantic key check.
        #
        # This uses the canonical schema function intentionally:
        # interval processing below is independently implemented,
        # while this verifies that the stored semantic key was
        # produced from the original annotation itself.
        stage_key = build_stage_key(skill, {})

        annotation_stage_keys.add(stage_key)

        for interval_idx, interval in enumerate(intervals):

            expected_raw_count += 1

            raw_start = int(interval[0])
            raw_end = int(interval[1])

            identity = (
                dataset_index,
                episode_id,
                skill_idx,
                interval_idx,
            )

            if identity not in raw_by_identity:
                raw_errors.append(
                    (
                        "missing_raw_row",
                        identity,
                    )
                )
                continue

            row = raw_by_identity[identity]

            # ----------------------------------------------
            # Reproduce source bounds policy independently
            # ----------------------------------------------

            reversed_bounds = raw_start > raw_end

            start = min(raw_start, raw_end)

            end = min(
                max(raw_start, raw_end),
                int(ep["source_num_frames"]),
            )

            if not (
                0 <= start
                < end
                <= int(ep["source_num_frames"])
            ):
                raw_errors.append(
                    (
                        "annotation_has_no_usable_span",
                        identity,
                        raw_start,
                        raw_end,
                    )
                )
                continue

            clipped = (
                max(raw_start, raw_end)
                > int(ep["source_num_frames"])
            )

            # ----------------------------------------------
            # Compare raw asset
            # ----------------------------------------------

            checks = {
                "episode_id": episode_id,
                "dataset_index": dataset_index,
                "task_id": int(ep["task_id"]),
                "task_name": ep["task_name"],
                "skill_idx": skill_idx,
                "interval_idx": interval_idx,
                "raw_segment_id":
                    f"{skill_idx}:{interval_idx}",
                "raw_start_frame": raw_start,
                "raw_end_frame": raw_end,
                "start_frame": start,
                "end_frame": end,
                "is_invalid": reversed_bounds,
                "invalid_reason":
                    "reversed_bounds"
                    if reversed_bounds
                    else None,
                "source_interval_adjustment":
                    "clip_to_source_end"
                    if clipped
                    else None,
                "current_stage_key": stage_key,
                "current_stage_text":
                    stage_text(stage_key),
            }

            for field, expected in checks.items():

                actual = row.get(field)

                if actual != expected:
                    raw_errors.append(
                        (
                            "raw_field_mismatch",
                            identity,
                            field,
                            actual,
                            expected,
                        )
                    )

            if reversed_bounds:
                expected_invalid.append(identity)

            if clipped:
                expected_clipped.append(identity)


extra_raw = sorted(
    set(raw_by_identity)
    - {
        (
            int(ep.get("dataset_index", 0)),
            str(ep["episode_id"]),
            int(skill.get("skill_idx", idx)),
            interval_idx,
        )
        for ep in episodes
        for idx, skill in enumerate(
            json.loads(
                Path(ep["annotation_path"]).read_text()
            )["skill_annotation"]
        )
        for interval_idx, _ in enumerate(
            parse_annotation_intervals(
                skill.get("frame_duration")
            )
        )
    }
)


print("----- Annotation -> raw asset -----")
print("expected raw rows :", expected_raw_count)
print("actual raw rows   :", len(raw))
print("invalid expected  :", len(expected_invalid))
print("clipped expected  :", len(expected_clipped))
print("raw errors        :", len(raw_errors))
print("extra raw rows    :", len(extra_raw))

if raw_errors:
    print("\nFirst raw errors:")
    for e in raw_errors[:30]:
        print(e)

if extra_raw:
    print("\nFirst extra raw identities:")
    for e in extra_raw[:20]:
        print(e)

assert expected_raw_count == len(raw)
assert not raw_errors
assert not extra_raw

assert len(expected_invalid) == config[
    "num_invalid_raw_intervals"
]

assert len(expected_clipped) == config[
    "num_clipped_raw_intervals"
]

print("[PASS] every original annotation interval")
print("       maps exactly to one raw segment")
print(
    "[PASS] invalid raw intervals:",
    len(expected_invalid),
)
print(
    "[PASS] clipped raw intervals:",
    len(expected_clipped),
)
print()


# ============================================================
# 3. Vocabulary source fidelity
# ============================================================

# Train-only vocabulary should equal all observed training
# canonical stage keys plus DONE.

expected_vocab_keys = {
    DONE_KEY,
    *annotation_stage_keys,
}

actual_vocab_keys = set(vocab)

missing_vocab = expected_vocab_keys - actual_vocab_keys
extra_vocab = actual_vocab_keys - expected_vocab_keys

print("----- Vocabulary -----")
print("expected classes :", len(expected_vocab_keys))
print("actual classes   :", len(actual_vocab_keys))
print("missing vocab    :", len(missing_vocab))
print("extra vocab      :", len(extra_vocab))

assert not missing_vocab
assert not extra_vocab
assert vocab[DONE_KEY] == DONE_ID
assert sorted(vocab.values()) == list(
    range(len(vocab))
)

print("[PASS] vocabulary exactly matches annotation stages")
print()


# ============================================================
# 4. Independently rebuild raw -> exclusive final partitions
# ============================================================

raw_by_episode = defaultdict(list)

for row in raw:
    key = ep_key(
        row.get("dataset_index", 0),
        row["episode_id"],
    )
    raw_by_episode[key].append(row)


expected_final = []
expected_conflicts = []


for key, raw_rows in raw_by_episode.items():

    by_id = {
        row["raw_segment_id"]: row
        for row in raw_rows
    }

    starts = defaultdict(set)
    ends = defaultdict(set)

    for row in raw_rows:
        rid = row["raw_segment_id"]
        starts[int(row["start_frame"])].add(rid)
        ends[int(row["end_frame"])].add(rid)

    boundaries = sorted(
        set(starts) | set(ends)
    )

    active = set()
    rows = []

    for left, right in zip(
        boundaries,
        boundaries[1:],
    ):

        # half-open semantics:
        # segments ending at left are inactive;
        # segments starting at left are active.
        active.difference_update(
            ends.get(left, set())
        )

        active.update(
            starts.get(left, set())
        )

        if not active or left >= right:
            continue

        ids = tuple(sorted(active))

        # Merge adjacent regions that have the same source set.
        if (
            rows
            and rows[-1]["end_frame"] == left
            and tuple(
                rows[-1]["raw_segment_ids"]
            ) == ids
        ):
            rows[-1]["end_frame"] = right
            continue

        source = by_id[ids[0]]

        ambiguous = (
            len(ids) != 1
            or any(
                bool(by_id[rid]["is_invalid"])
                for rid in ids
            )
        )

        row = {
            "episode_id":
                str(source["episode_id"]),
            "dataset_index":
                int(source.get("dataset_index", 0)),
            "split":
                source["split"],
            "task_id":
                int(source["task_id"]),
            "task_name":
                source["task_name"],
            "start_frame":
                int(left),
            "end_frame":
                int(right),
            "raw_segment_ids":
                list(ids),
            "is_ambiguous":
                bool(ambiguous),
        }

        if not ambiguous:
            row.update(
                skill_idx=int(
                    source["skill_idx"]
                ),
                interval_idx=int(
                    source["interval_idx"]
                ),
                progress_start_frame=int(
                    source["start_frame"]
                ),
                progress_end_frame=int(
                    source["end_frame"]
                ),
                current_stage_key=
                    source[
                        "current_stage_key"
                    ],
                current_stage_text=
                    source[
                        "current_stage_text"
                    ],
            )
        else:
            row.update(
                skill_idx=-1,
                interval_idx=-1,
                progress_start_frame=int(left),
                progress_end_frame=int(right),
                current_stage_key=None,
                current_stage_text="",
            )

        rows.append(row)

    # --------------------------------------------------------
    # Next-stage generation
    # --------------------------------------------------------

    for i, row in enumerate(rows):

        following = (
            rows[i + 1]
            if i + 1 < len(rows)
            else None
        )

        next_key = (
            following["current_stage_key"]
            if following is not None
            else DONE_KEY
        )

        if row["is_ambiguous"]:
            next_key = None

        row["next_stage_key"] = next_key

        row["next_stage_text"] = (
            stage_text(next_key)
            if next_key is not None
            else ""
        )

        row["is_terminal_stage"] = (
            following is None
            and not row["is_ambiguous"]
        )

        row["current_stage_id"] = (
            vocab.get(
                row["current_stage_key"],
                IGNORE_ID,
            )
        )

        row["next_stage_id"] = (
            vocab.get(
                row["next_stage_key"],
                IGNORE_ID,
            )
        )

        row["segment_idx"] = i

        if row["is_ambiguous"]:

            sources = [
                by_id[rid]
                for rid
                in row["raw_segment_ids"]
            ]

            expected_conflicts.append(
                {
                    "episode_key": key,
                    "episode_id":
                        row["episode_id"],
                    "task_id":
                        row["task_id"],
                    "start_frame":
                        row["start_frame"],
                    "end_frame":
                        row["end_frame"],
                    "raw_segment_ids":
                        row[
                            "raw_segment_ids"
                        ],
                    "skill_indices":
                        [
                            int(s["skill_idx"])
                            for s in sources
                        ],
                    "stage_keys":
                        [
                            s[
                                "current_stage_key"
                            ]
                            for s in sources
                        ],
                    "reason":
                        (
                            "invalid_source_interval"
                            if any(
                                bool(
                                    s[
                                        "is_invalid"
                                    ]
                                )
                                for s in sources
                            )
                            else
                            "overlapping_intervals"
                        ),
                }
            )

    expected_final.extend(rows)


expected_final.sort(
    key=lambda r: (
        int(r["dataset_index"]),
        str(r["episode_id"]),
        int(r["start_frame"]),
    )
)

actual_final = sorted(
    final,
    key=lambda r: (
        int(r.get("dataset_index", 0)),
        str(r["episode_id"]),
        int(r["start_frame"]),
    )
)


# ============================================================
# 5. Compare independently reconstructed final asset
# ============================================================

final_errors = []

if len(expected_final) != len(actual_final):
    final_errors.append(
        (
            "final_count",
            len(actual_final),
            len(expected_final),
        )
    )


COMPARE_FIELDS = [
    "episode_id",
    "dataset_index",
    "split",
    "task_id",
    "task_name",
    "segment_idx",
    "start_frame",
    "end_frame",
    "raw_segment_ids",
    "is_ambiguous",
    "skill_idx",
    "interval_idx",
    "progress_start_frame",
    "progress_end_frame",
    "current_stage_key",
    "current_stage_text",
    "next_stage_key",
    "next_stage_text",
    "is_terminal_stage",
    "current_stage_id",
    "next_stage_id",
]


for i, (expected, actual) in enumerate(
    zip(expected_final, actual_final)
):

    for field in COMPARE_FIELDS:

        e = expected.get(field)
        a = actual.get(field)

        # Normalize lists that may come back as numpy/list objects.
        if field == "raw_segment_ids":
            e = list(e or [])
            a = list(a or [])

        if a != e:
            final_errors.append(
                (
                    "final_field_mismatch",
                    i,
                    expected.get(
                        "episode_id"
                    ),
                    field,
                    a,
                    e,
                )
            )


print("----- Raw -> final partition -----")
print(
    "expected final segments :",
    len(expected_final),
)
print(
    "actual final segments   :",
    len(actual_final),
)
print(
    "final mismatches        :",
    len(final_errors),
)

if final_errors:
    print("\nFirst final errors:")
    for e in final_errors[:30]:
        print(e)

assert not final_errors

print("[PASS] independent interval partition")
print("[PASS] ambiguous regions")
print("[PASS] progress original-source bounds")
print("[PASS] current/next stage IDs")
print()


# ============================================================
# 6. Conflict audit
# ============================================================

def conflict_identity(c):
    return (
        str(c["episode_id"]),
        int(c["start_frame"]),
        int(c["end_frame"]),
        tuple(c["raw_segment_ids"]),
    )


expected_conflict_map = {
    conflict_identity(c): c
    for c in expected_conflicts
}

actual_conflict_map = {
    conflict_identity(c): c
    for c in conflicts_file
}

missing_conflicts = (
    set(expected_conflict_map)
    - set(actual_conflict_map)
)

extra_conflicts = (
    set(actual_conflict_map)
    - set(expected_conflict_map)
)

expected_conflict_frames = sum(
    int(c["end_frame"])
    - int(c["start_frame"])
    for c in expected_conflicts
)


print("----- Conflicts -----")
print(
    "expected conflict intervals :",
    len(expected_conflicts),
)
print(
    "asset conflict intervals    :",
    len(conflicts_file),
)
print(
    "expected conflict frames    :",
    expected_conflict_frames,
)
print(
    "config conflict frames      :",
    config["num_conflict_frames"],
)

assert not missing_conflicts
assert not extra_conflicts

assert len(expected_conflicts) == config[
    "num_conflict_intervals"
]

assert expected_conflict_frames == config[
    "num_conflict_frames"
]

print(
    "[PASS] conflict intervals:",
    len(expected_conflicts),
)

print(
    "[PASS] conflict frames:",
    expected_conflict_frames,
)

print()


# ============================================================
# 7. Independent training-safe next-stage audit
# ============================================================

rows_by_episode = defaultdict(list)

for row in actual_final:
    rows_by_episode[
        ep_key(
            row.get("dataset_index", 0),
            row["episode_id"],
        )
    ].append(row)


independent_masked_next = 0
independent_safe_next = 0


for key, rows in rows_by_episode.items():

    rows.sort(
        key=lambda r: int(
            r["start_frame"]
        )
    )

    for index, row in enumerate(rows):

        following = (
            rows[index + 1]
            if index + 1 < len(rows)
            else None
        )

        if following is None:
            safe = (
                bool(
                    row.get(
                        "is_terminal_stage",
                        False,
                    )
                )
                and int(
                    row["next_stage_id"]
                )
                == DONE_ID
            )
        else:
            safe = (
                int(row["end_frame"])
                == int(
                    following[
                        "start_frame"
                    ]
                )
                and not bool(
                    following.get(
                        "is_ambiguous",
                        False,
                    )
                )
                and int(
                    following[
                        "current_stage_id"
                    ]
                )
                > 0
                and int(
                    row[
                        "next_stage_id"
                    ]
                )
                == int(
                    following[
                        "current_stage_id"
                    ]
                )
            )

        safe = (
            safe
            and not bool(
                row.get(
                    "is_ambiguous",
                    False,
                )
            )
            and int(
                row[
                    "current_stage_id"
                ]
            )
            > 0
        )

        if (
            int(row["next_stage_id"])
            != IGNORE_ID
            and not safe
        ):
            independent_masked_next += 1

        elif (
            int(row["next_stage_id"])
            != IGNORE_ID
            and safe
        ):
            independent_safe_next += 1


# Now compare to the actual training implementation.
lookup = StageSupervisionLookup.from_assets(
    ASSETS,
    split="train",
    use_task_stage_mask=True,
)

actual_masked_next = int(
    lookup.target_policy_report[
        "masked_next_segments"
    ]
)


print("----- Training-safe next targets -----")
print(
    "policy:",
    TRAINING_TARGET_POLICY,
)

print(
    "independent masked next :",
    independent_masked_next,
)

print(
    "training lookup masked  :",
    actual_masked_next,
)

print(
    "safe stored next        :",
    independent_safe_next,
)

assert (
    independent_masked_next
    == actual_masked_next
)

print("[PASS] training next-stage masking")
print()


# ============================================================
# 8. Exceptional interval audit file consistency
# ============================================================

# stage_invalid_intervals.json intentionally contains BOTH:
#   1. invalid/reversed raw intervals
#   2. clipped/adjusted raw intervals
#
# Do not assume the two sets are disjoint.
expected_exceptional = (
    set(expected_invalid)
    | set(expected_clipped)
)

actual_exceptional = {}

for row in invalid_file:
    identity = (
        int(row.get("dataset_index", 0)),
        str(row["episode_id"]),
        int(row["skill_idx"]),
        int(row["interval_idx"]),
    )

    if identity in actual_exceptional:
        raise RuntimeError(
            f"Duplicate exceptional interval in audit JSON: {identity}"
        )

    actual_exceptional[identity] = row


actual_ids = set(actual_exceptional)

missing_exceptional = (
    expected_exceptional - actual_ids
)

extra_exceptional = (
    actual_ids - expected_exceptional
)


print("----- Exceptional intervals -----")

print(
    "independent invalid/reversed :",
    len(expected_invalid),
)

print(
    "independent clipped          :",
    len(expected_clipped),
)

print(
    "expected exceptional union   :",
    len(expected_exceptional),
)

print(
    "audit JSON rows              :",
    len(invalid_file),
)

print(
    "missing exceptional rows     :",
    len(missing_exceptional),
)

print(
    "extra exceptional rows       :",
    len(extra_exceptional),
)


if missing_exceptional:
    print(
        "first missing:",
        sorted(missing_exceptional)[:20],
    )

if extra_exceptional:
    print(
        "first extra:",
        sorted(extra_exceptional)[:20],
    )


assert not missing_exceptional
assert not extra_exceptional
assert len(invalid_file) == len(expected_exceptional)

print(
    "[PASS] exceptional interval audit file "
    "(invalid OR clipped)"
)
print()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("================================================")
print("OUC ASSET GENERATION AUDIT: PASS")
print("================================================")

print(
    f"episodes                    : {len(episodes)}"
)
print(
    f"raw intervals               : {len(raw)}"
)
print(
    f"final segments              : {len(final)}"
)
print(
    f"stage classes               : {len(vocab)}"
)
print(
    f"invalid raw intervals       : {len(expected_invalid)}"
)
print(
    f"clipped raw intervals       : {len(expected_clipped)}"
)
print(
    f"conflict intervals          : {len(expected_conflicts)}"
)
print(
    f"conflict frames             : {expected_conflict_frames}"
)
print(
    f"masked unsafe next segments : {independent_masked_next}"
)

print()
print("Verified chain:")
print(
    "annotation JSON"
    " -> raw_stage_segments"
    " -> exclusive partition"
    " -> stage_segments"
    " -> vocabulary IDs"
    " -> safe next-stage targets"
)

print()
print("RESULT: PASS")
