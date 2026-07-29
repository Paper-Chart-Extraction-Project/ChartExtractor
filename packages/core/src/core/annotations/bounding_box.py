"""Contains the BoundingBox class."""

from core.annotations import AnnotationId
from core.geometry import Rectangle
from pydantic import BaseModel, ConfigDict, Field
from uuid import uuid4


class BoundingBox(BaseModel):
    """A rectangle with a classification.

    Attributes:
        rectangle (Rectangle):
            The rectangle that this BoundingBox encloses.
        category (str):
            The BoundingBox's category. Cannot be empty.
        annotation_id (AnnotationId):
            The annotations unique identifier.
    """

    model_config = ConfigDict(frozen=True)

    rectangle: Rectangle
    category: str = Field(min_length=1)
    annotation_id: AnnotationId = Field(default_factory=uuid4)
