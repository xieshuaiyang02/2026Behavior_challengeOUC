"""Compatibility imports for LeRobot dataset APIs."""

from typing import Any

try:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets import MultiLeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

__all__ = ["LeRobotDataset", "LeRobotDatasetMetadata", "MultiLeRobotDataset", "tasks_from_metadata"]


def tasks_from_metadata(metadata: Any) -> dict[int, str]:
    tasks = metadata.tasks
    if isinstance(tasks, dict):
        return tasks
    # New LeRobot DataFrame, index is task name, task_index is in column
    import pandas as pd
    if isinstance(tasks, pd.DataFrame):
        return {int(row["task_index"]): str(task_name) for task_name, row in tasks.iterrows()}
    raise TypeError(f"Unsupported tasks type: {type(tasks)}")