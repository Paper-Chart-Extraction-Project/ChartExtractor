"""Contains the BoundingBox class."""

from core.geometry.point import Point
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self


class BoundingBox(BaseModel):
    """Represents a rectangle drawn around an object on an image.

    Immutable, should be changed by creating new BoundingBoxes.

    Attributes:
        left (float):
            The left side of the bounding box.
        top (float):
            The top side of the bounding box.
        right (float):
            The right side of the bounding box.
        bottom (float):
            The bottom side of the bounding box.
        center (Point):
            The center of the bounding box.
        width (float):
            The width of the bounding box.
        height (float):
            The height of the bounding box.
        area (float):
            The area of the bounding box.

    Private Attributes:
        _top_left (Point):
            The top left point of the bounding box.
        _bottom_right (Point):
            The top left point of the bounding box.
    """

    model_config = ConfigDict(frozen=True)

    _top_left: Point
    _bottom_right: Point
