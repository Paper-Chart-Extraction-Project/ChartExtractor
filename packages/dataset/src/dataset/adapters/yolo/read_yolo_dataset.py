"""A module containing the main function of the yolo adapter subpackage: read_yolo_dataset."""

from pathlib import Path
from typing import Literal

from dataset import VisionDataset

from .yolo_detect_dataset import _YoloDetectDataset
from .yolo_pose_dataset import _YoloPoseDataset


def read_yolo_dataset(
    folderpath: Path,
    yaml_file_name: str = "dataset.yaml",
    dataset_type: Literal["detect", "pose"] = "detect",
) -> VisionDataset:
    """Reads a YOLO dataset from a path to the dataset's root folder.

    Args:
        folderpath (Path):
            The path to the root of the yolo dataset's folder.
        yaml_file_name (str):
            The filename of the dataset yaml config file.
        detect_type (Literal["detect", "pose"]):
            The type of dataset to read.

    Returns:
        A VisionDataset containing the contents of the yolo dataset.
    """
    match dataset_type:
        case "detect":
            yolo_detect_dataset: _YoloDetectDataset = (
                _YoloDetectDataset.from_folderpath(
                    folderpath=folderpath,
                    yaml_file_name=yaml_file_name,
                )
            )
            return yolo_detect_dataset.read_to_vision_dataset()
        case "pose":
            yolo_pose_dataset: _YoloPoseDataset = _YoloPoseDataset.from_folderpath(
                folderpath=folderpath,
                yaml_file_name=yaml_file_name,
            )
            return yolo_pose_dataset.read_to_vision_dataset()
