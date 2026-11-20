"""A module that can read/write yolo detect datasets to DetectVisionDataset."""

from .yolo_dataset import _YoloDatasetConfig


class _YoloDetectDatasetConfig(_YoloDatasetConfig):
    """Reads, validates, and stores data from a yolo detect dataset.yaml file.

    Identical to the base class, but exists for typing reasons.
    """
