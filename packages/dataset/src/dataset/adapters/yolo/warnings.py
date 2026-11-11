"""A module containing generic warnings for yolo-specific dataset issues."""

from dataset import DatasetWarning


class YoloDatasetWarning(DatasetWarning):
    """A generic warning regarding yolo dataset issues."""


class RelativeDatasetPathWarning(YoloDatasetWarning):
    """A warning issued when the 'path' attribute of the dataset config is a relative path."""
