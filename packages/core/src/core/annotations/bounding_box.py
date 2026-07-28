"""Contains the BoundingBox class."""

from core.geometry import Rectangle
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """A rectangle with a classification.

    Attributes:
        rectangle (Rectangle):
            The rectangle that this BoundingBox encloses.
        category (str):
            The BoundingBox's category. Cannot be empty.
    """

    model_config = ConfigDict(frozen=True)

    rectangle: Rectangle
    category: str = Field(min_length=1)
