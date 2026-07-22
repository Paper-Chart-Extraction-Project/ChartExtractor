"""Contains the BoundingBox class."""

from core.geometry.point import Point
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self


class BoundingBox(BaseModel):
    """Represents a rectangle drawn around an object on an image.

    Immutable, should be changed by creating new BoundingBoxes.

    Attributes:
        top_left (Point):
            The top left point of the bounding box.
        bottom_right (Point):
            The top left point of the bounding box.
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
    """

    model_config = ConfigDict(frozen=True)

    top_left: Point
    bottom_right: Point

    @model_validator(mode="after")
    def check_top_left_higher_and_further_left_of_bottom_right(self) -> Self:
        """Ensures that the top left point is higher and further to the left of the bottom right."""
        if self.top_left.x > self.bottom_right.x:
            raise ValueError(
                f"Top left point of {self} is further right than its bottom right point."
            )
        if self.top_left.y > self.bottom_right.y:
            raise ValueError(
                f"Top left point of {self} is higher than its bottom right point."
            )

        return self

    @classmethod
    def from_center_xywh(
        cls, x_center: float, y_center: float, width: float, height: float
    ) -> "BoundingBox":
        """Creates a BoundingBox from the center, width, and height of the box.

        Args:
            x_center (float):
                The x coordinate of the center of the bounding box.
            y_center (float):
                The y coordinate of the center of the bounding box.
            width (float):
                The width of the bounding box.
            height (float):
                The height of the bounding box.

        Returns:
            A BoundingBox at the location supplied in the arguments.
        """
        left: float = x_center - (1 / 2) * width
        top: float = y_center - (1 / 2) * height
        right: float = x_center + (1 / 2) * width
        bottom: float = y_center + (1 / 2) * height

        top_left: Point = Point(x=left, y=top)
        bottom_right: Point = Point(x=right, y=bottom)

        return cls(top_left=top_left, bottom_right=bottom_right)
