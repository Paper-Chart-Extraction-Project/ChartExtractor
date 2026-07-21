"""Contains the Point class."""

from pydantic import BaseModel, ConfigDict


class Point(BaseModel):
    """A two-dimensional point in space.

    An immutable point in 2d.

    Attributes:
        x (float):
            The x value of the point.
        y (float):
            The y value of the point.
    """

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
