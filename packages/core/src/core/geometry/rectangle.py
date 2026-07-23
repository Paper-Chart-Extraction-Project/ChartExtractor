"""Contains the Rectangle class."""

from core.geometry import Point
import math
from pydantic import BaseModel, computed_field, ConfigDict, model_validator
from typing import Tuple
from typing_extensions import Self
import warnings


class Rectangle(BaseModel):
    """Represents a rectangle drawn around an object on an image.

    Assumes that the top left of the image is the (0, 0) coordinate. Or, in other words, that
    the x axis runs from left to right, and the y axis runs from top to bottom.
    Immutable, must be changed by creating new Rectangles.

    Attributes:
        top_left (Point):
            The top left point of the rectangle.
        bottom_right (Point):
            The bottom right point of the rectangle.
        left (float):
            The left side of the rectangle.
        top (float):
            The top side of the rectangle.
        right (float):
            The right side of the rectangle.
        bottom (float):
            The bottom side of the rectangle.
        center (Point):
            The center of the rectangle.
        width (float):
            The width of the rectangle.
        height (float):
            The height of the rectangle.
        area (float):
            The area of the rectangle.
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
    def warn_on_degenerate_rectangle(self) -> Self:
        """Warns the user if the Rectangle is degenerate (has an area of 0)."""
        left_right_degen: bool = math.isclose(self.left, self.right, rel_tol=1e-9)
        top_bottom_degen: bool = math.isclose(self.top, self.bottom, rel_tol=1e-9)

        if left_right_degen and top_bottom_degen:
            warnings.warn(
                f"{self} is a completely degenerate (area=0) Rectangle.", UserWarning
            )
        elif left_right_degen:
            warnings.warn(
                f"{self} is a left-right degenerate (area=0) Rectangle.", UserWarning
            )
        elif top_bottom_degen:
            warnings.warn(
                f"{self} is a top-bottom degenerate (area=0) Rectangle.", UserWarning
            )

        return self

    @computed_field
    @property
    def left(self) -> float:
        """The left of the rectangle."""
        return self.top_left.x

    @computed_field
    @property
    def top(self) -> float:
        """The top of the rectangle."""
        return self.top_left.y

    @computed_field
    @property
    def right(self) -> float:
        """The right of the rectangle."""
        return self.bottom_right.x

    @computed_field
    @property
    def bottom(self) -> float:
        """The bottom of the rectangle."""
        return self.bottom_right.y

    @computed_field
    @property
    def width(self) -> float:
        """The width of the rectangle."""
        return self.right - self.left

    @computed_field
    @property
    def height(self) -> float:
        """The height of the rectangle."""
        return self.bottom - self.top

    @computed_field
    @property
    def area(self) -> float:
        """The area of the rectangle."""
        return self.width * self.height

    @computed_field
    @property
    def top_right(self) -> Point:
        """The top right of the rectangle."""
        return Point(x=self.right, y=self.top)

    @computed_field
    @property
    def bottom_left(self) -> Point:
        """The bottom left of the rectangle."""
        return Point(x=self.left, y=self.bottom)

    @computed_field
    @property
    def center(self) -> Point:
        """The center of the rectangle."""
        return Point(
            x=(1 / 2) * (self.left + self.right), y=(1 / 2) * (self.top + self.bottom)
        )

    @classmethod
    def from_center_xywh(
        cls, x_center: float, y_center: float, width: float, height: float
    ) -> "Rectangle":
        """Creates a Rectangle based on a center, width, and height.

        Args:
            x_center (float):
                The x coordinate of the center of the rectangle.
            y_center (float):
                The y coordinate of the center of the rectangle.
            width (float):
                The width of the rectangle.
            height (float):
                The height of the rectangle.

        Returns:
            A Rectangle at the location supplied in the arguments.
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
    ) -> "Rectangle":
        """Creates a Rectangle based on a top, left, width, and height.

        Note the order of the arguments.

        Args:
            left (float):
                The x coordinate of the left of the rectangle.
            top (float):
                The y coordinate of the top of the rectangle.
            width (float):
                The width of the rectangle.
            height (float):
                The height of the rectangle.

        Returns:
            A Rectangle at the location supplied in the arguments.

        Raises:
            ValueError:
                If width or height is negative.
        """
        if width < 0 or height < 0:
            raise ValueError(
                f"Width and height must be non-negative (width={width}, height={height})."
            )
        right: float = left + width
        bottom: float = top + height

        top_left: Point = Point(x=left, y=top)
        bottom_right: Point = Point(x=right, y=bottom)

        return cls(top_left=top_left, bottom_right=bottom_right)

    @classmethod
    def from_left_top_right_bottom(
        cls, left: float, top: float, right: float, bottom: float
    ) -> "Rectangle":
        """Creates a Rectangle based on a left, top, right, and bottom.

        Args:
            left (float):
                The x coordinate of the left of the rectangle.
            top (float):
                The y coordinate of the top of the rectangle.
            right (float):
                The x coordinate of the right of the rectangle.
            bottom (float):
                The y coordinate of the bottom of the rectangle.

        Returns:
            A Rectangle at the location supplied in the arguments.
        """
        top_left: Point = Point(x=left, y=top)
        bottom_right: Point = Point(x=right, y=bottom)

        return cls(top_left=top_left, bottom_right=bottom_right)

    def to_center_xywh(self) -> Tuple[float, float, float, float]:
        """Returns this Rectangle as a tuple containing the center, width, and height.

        Returns:
            A tuple containing the rectangle's (x center, y center, width, height).
        """
        return (self.center.x, self.center.y, self.width, self.height)

    def to_top_left_xywh(self) -> Tuple[float, float, float, float]:
        """Returns this Rectangle as a tuple containing the left, top, width, and height.

        Note the order of the return is left, top, width, height, not top, left ...

        Returns:
            A tuple containing the rectangle's (left, top, width, the height).
        """
        return (self.left, self.top, self.width, self.height)

    def to_left_top_right_bottom(self) -> Tuple[float, float, float, float]:
        """Returns this Rectangle as a tuple containing the left, top, right, bottom.

        Returns:
            A tuple containing the rectangle's (left, top, right, bottom).
        """
        return (self.left, self.top, self.right, self.bottom)

    def contains_point(self, p: Point) -> bool:
        """Whether or not this rectangle contains the point p.

        If a point lies on the border of a bounding box, it is considered inside.

        Args:
            p (Point):
                The point in question.

        Returns:
            Whether or not this rectangle contains the point p, or whether p is on the border
            of this rectangle.
        """
        return all(
            [self.left <= p.x, self.right >= p.x, self.top <= p.y, self.bottom >= p.y]
        )
