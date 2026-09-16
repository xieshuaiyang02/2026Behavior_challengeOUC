"""BEHAVIOR evaluation with an atomically overwritten live camera preview.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval_ouc \
        --task-name turning_on_radio \
        --robot-config omnigibson/eval/r1pro.yaml \
        --mode public_test \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval --write-video
"""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from omegaconf import OmegaConf

from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def save_live_frame(evaluator, path: Path, *, task: str, instance: int, rollout: int, step: int) -> None:
    """Save current observation cameras without advancing or rendering the simulator."""
    def camera_rgb(role, size):
        camera_name = evaluator.robot_camera_names.get(role)
        key = f"{camera_name}::rgb"
        if camera_name is None or key not in evaluator.obs:
            raise ValueError(f"Live preview requires the {role} RGB camera in evaluator.obs")
        pixels = evaluator.obs[key]
        if hasattr(pixels, "detach"):
            pixels = pixels.detach().cpu().numpy()
        return cv2.resize(np.asarray(pixels)[..., :3], size)

    left = camera_rgb("left_wrist", (224, 224))
    right = camera_rgb("right_wrist", (224, 224))
    head = camera_rgb("head", (448, 448))
    rgb = np.hstack([np.vstack([left, right]), head])
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.copyMakeBorder(frame, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for y, text in ((18, task), (38, f"instance={instance} rollout={rollout} step={step}")):
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode live evaluation frame")
    # Same directory guarantees replace is atomic, even when output is on another filesystem.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".latest-", suffix=".jpg", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded.tobytes())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help=(
            "Optional path to YAML/JSON file containing one complete robot config dictionary with canonical "
            "'model' and 'name' fields. Add eval.camera_sensor_names to configure eval camera roles."
        ),
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Instance indices for the selected mode. For train these are direct train instance IDs; "
            "for public_test / hidden_test these index into that 20-instance split."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "public_test", "hidden_test"),
        default="public_test",
        help="Instance split to evaluate. Default: public_test.",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 1.5x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument(
        "--policy",
        choices=("websocket", "local"),
        default="websocket",
        help="Policy backend to use. local emits zero actions and is intended for eval smoke tests.",
    )
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--live-frame", action=argparse.BooleanOptionalAction, default=True,
        help="Overwrite <output-dir>/latest.jpg with current head + wrist images (default: enabled).",
    )
    parser.add_argument(
        "--live-frame-interval", type=int, default=1,
        help="Save a live preview every N environment steps; initial and final frames are always saved.",
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an MP4 rollout video (head + wrist cameras) per rollout under <output-dir>/videos.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for saved rollout videos.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    args = parser.parse_args()
    if args.live_frame_interval < 1:
        parser.error("--live-frame-interval must be at least 1")
    return args


def main() -> None:
    args = parse_args()

    gm.HEADLESS = args.headless

    seed = seed_everything(DEFAULT_EVAL_SEED)
    logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    logger.info(f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}")

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
        }
    else:
        model_cfg = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "mode": args.mode,
            "seed": seed,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    live_frame_path = Path(args.output_dir).expanduser() / "latest.jpg"
    if args.live_frame:
        logger.info(f"Live camera preview (overwritten during evaluation): {live_frame_path.resolve()}")
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    results = []
    with Evaluator(cfg) as evaluator:
        for instance_id in instance_ids:
            try:
                evaluator.reset()
                evaluator.load_task_instance(int(instance_id))
            except Exception:
                logger.exception(f"Failed to load task instance {instance_id}.")
                raise
            for rollout_id in range(args.num_rollouts):
                video_path = os.path.join(video_dir, f"{args.task_name}_{instance_id}_{rollout_id}.mp4")
                try:
                    evaluator.reset()
                    if args.live_frame:
                        save_live_frame(
                            evaluator, live_frame_path, task=args.task_name,
                            instance=int(instance_id), rollout=rollout_id, step=0,
                        )
                    if args.write_video:
                        evaluator.start_recording(video_path, rate=args.video_fps)
                    terminated = truncated = False
                    steps = 0
                    while not (terminated or truncated):
                        terminated, truncated = evaluator.step()
                        steps += 1
                        if args.live_frame and (steps % args.live_frame_interval == 0 or terminated or truncated):
                            save_live_frame(
                                evaluator, live_frame_path, task=args.task_name,
                                instance=int(instance_id), rollout=rollout_id, step=steps,
                            )

                    success = bool(evaluator.env.task.success)
                    metrics = {}
                    for metric in evaluator.metrics:
                        metrics.update(metric.aggregate(evaluator.env))

                    result = {
                        "task": args.task_name,
                        "instance_id": int(instance_id),
                        "rollout_id": rollout_id,
                        "steps": steps,
                        "success": success,
                        **metrics,
                    }
                    out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    video_msg = f" | video -> {video_path}" if args.write_video else ""
                    logger.info(
                        f"Result: instance={instance_id} rollout={rollout_id} steps={steps} "
                        f"success={success} q_score={q_score} -> {out_path}{video_msg}"
                    )
                    results.append(result)
                except Exception:
                    logger.exception(f"Instance {instance_id} rollout {rollout_id} failed.")
                    raise
                finally:
                    if args.write_video:
                        evaluator.stop_recording()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
    logger.info(f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")


if __name__ == "__main__":
    main()
