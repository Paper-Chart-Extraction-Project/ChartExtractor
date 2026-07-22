"""Contains the BoundingBox class."""

from core.geometry import Point
import math
from pydantic import BaseModel, computed_field, ConfigDict, model_validator
from typing import Optional
from typing_extensions import Dict, Self, Tuple
import warnings


class BoundingBox(BaseModel):
    """Represents a rectangle drawn around an object on an image.

    Assumes that the top left of the image is the (0, 0) coordinate. Or, in other words, that
    the x axis runs from left to right, and the y axis runs from top to bottom.
    Immutable, must be changed by creating new BoundingBoxes.

    Attributes:
        top_left (Point):
            The top left point of the bounding box.
        bottom_right (Point):
            The bottom right point of the bounding box.
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
        top_left_is_further_left: bool = self.top_left.x <= self.bottom_right.x
        top_left_is_higher: bool = self.top_left.y <= self.bottom_right.y

        if not top_left_is_further_left and not top_left_is_higher:
            raise ValueError(
                f"Top left point of {self} is lower and further right than its bottom right point."
            )
        if not top_left_is_further_left:
            raise ValueError(
                f"Top left point of {self} is further right than its bottom right point."
            )
        if not top_left_is_higher:
            raise ValueError(
                f"Top left point of {self} is lower than its bottom right point."
            )

        return self

    @model_validator(mode="after")
    def warn_on_degenerate_bounding_box(self) -> Self:
        """Warns the user if the BoundingBox is degenerate (has an area of 0)."""
        left_right_degen: bool = math.isclose(self.left, self.right, rel_tol=1e-9)
        top_bottom_degen: bool = math.isclose(self.top, self.bottom, rel_tol=1e-9)

        if left_right_degen and top_bottom_degen:
            warnings.warn(
                f"{self} is a completely degenerate (area=0) BoundingBox.", UserWarning
            )
        elif left_right_degen:
            warnings.warn(
                f"{self} is a left-right degenerate (area=0) BoundingBox.", UserWarning
            )
        elif top_bottom_degen:
            warnings.warn(
                f"{self} is a top-bottom degenerate (area=0) BoundingBox.", UserWarning
            )

        return self

    @computed_field
    @property
    def left(self) -> float:
        """The left of the bounding box."""
        return self.top_left.x

    @computed_field
    @property
    def top(self) -> float:
        """The top of the bounding box."""
        return self.top_left.y

    @computed_field
    @property
    def right(self) -> float:
        """The right of the bounding box."""
        return self.bottom_right.x

    @computed_field
    @property
    def bottom(self) -> float:
        """The bottom of the bounding box."""
        return self.bottom_right.y

    @computed_field
    @property
    def width(self) -> float:
        """The width of the bounding box."""
        return self.right - self.left

    @computed_field
    @property
    def height(self) -> float:
        """The height of the bounding box."""
        return self.bottom - self.top

    @computed_field
    @property
    def area(self) -> float:
        """The area of the bounding box."""
        return self.width * self.height

    @computed_field
    @property
    def top_right(self) -> Point:
        """The top right of the bounding box."""
        return Point(x=self.right, y=self.top)

    @computed_field
    @property
    def bottom_left(self) -> Point:
        """The bottom left of the bounding box."""
        return Point(x=self.left, y=self.bottom)

    @computed_field
    @property
    def center(self) -> Point:
        """The center of the bounding box."""
        return Point(
            x=(1 / 2) * (self.left + self.right), y=(1 / 2) * (self.top + self.bottom)
        )

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

    @classmethod
    def from_top_left_xywh(
        cls, left: float, top: float, width: float, height: float
    ) -> "BoundingBox":
        """Creates a BoundingBox from the top, left, width, and height of the box.

        Note the order of the arguments.

        Args:
            left (float):
                The x coordinate of the left of the bounding box.
            top (float):
                The y coordinate of the top of the bounding box.
            width (float):
                The width of the bounding box.
            height (float):
                The height of the bounding box.

        Returns:
            A BoundingBox at the location supplied in the arguments.
        """
        right: float = left + width
        bottom: float = top + height

        top_left: Point = Point(x=left, y=top)
        bottom_right: Point = Point(x=right, y=bottom)

        return cls(top_left=top_left, bottom_right=bottom_right)

    @classmethod
    def from_left_top_right_bottom(
        cls, left: float, top: float, right: float, bottom: float
    ) -> "BoundingBox":
        """Creates a BoundingBox from the left, top, right, and bottom of the box.

        Args:
            left (float):
                The x coordinate of the left of the bounding box.
            top (float):
                The y coordinate of the top of the bounding box.
            right (float):
                The x coordinate of the right of the bounding box.
            bottom (float):
                The y coordinate of the bottom of the bounding box.

        Returns:
            A BoundingBox at the location supplied in the arguments.
        """
        top_left: Point = Point(x=left, y=top)
        bottom_right: Point = Point(x=right, y=bottom)

        return cls(top_left=top_left, bottom_right=bottom_right)

    @classmethod
    def from_yolo(
        cls,
        yolo_line: str,
        image_width: Optional[int],
        image_height: Optional[int],
        id_to_category_map: Optional[Dict[int, str]],
    ) -> "BoundingBox":
        """Creates a BoundingBox from a line in a yolo labels file.

        Args:
            yolo_line (str):
                A single line from a yolo labels file.
            image_width (Optional[int]):
                The image's width that the label is for. If not supplied, the values are left as
                the image-normalized values.
            image_height (Optional[int]):
                The image's height that the label is for. If not supplied, the values are left as
                the image-normalized values.
            id_to_category_map (Optional[Dict[int, str]]):
                The mapping from the numbered category to the name of the category. If not
                supplied, the number category is converted directly to a string
                (eg: 5 -> "5").
        """
        raise NotImplementedError()

    def to_center_xywh(self) -> Tuple[float, float, float, float]:
        """Returns this BoundingBox as a tuple containing the center, width, and height.

        Returns:
            A tuple containing the bbox's (x center, y center, width, height).
        """
        return (self.center.x, self.center.y, self.width, self.height)

    def to_top_left_xywh(self) -> Tuple[float, float, float, float]:
        """Returns this BoundingBox as a tuple containing the left, top, width, and height.

        Note the order of the return is left, top, width, height, not top, left ...

        Returns:
            A tuple containing the bbox's (left, top, width, the height).
        """
        return (self.left, self.top, self.width, self.height)

    def to_left_top_right_bottom(self) -> Tuple[float, float, float, float]:
        """Returns this BoundingBox as a tuple containing the left, top, right, bottom.

        Returns:
            A tuple containing the bbox's (left, top, right, bottom).
        """
        return (self.left, self.top, self.right, self.bottom)
