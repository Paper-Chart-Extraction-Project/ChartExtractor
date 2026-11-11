"""A module containing custom exceptions for yolo-specific dataset errors."""

from dataset import DatasetError


class YoloDatasetError(DatasetError):
    """A generic exception regarding yolo dataset issues."""


class YoloDatasetConfigError(YoloDatasetError):
    """An error raised when there is an issue with the yolo dataset config file."""


class YoloDatasetStructureError(YoloDatasetError):
    """An error raised when there is an issue with the folder structure of the yolo dataset."""


class YoloAnnotationError(YoloDatasetError):
    """An error raised when there is an issue parsing a yolo annotation."""
