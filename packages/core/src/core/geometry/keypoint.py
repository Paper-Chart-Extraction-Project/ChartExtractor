"""Contains the Keypoint class."""
from enum import StrEnum


class VisibilityStatus(StrEnum):
    """An enum with three variants based on a keypoint's visibility."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    MISSING = "missing"

