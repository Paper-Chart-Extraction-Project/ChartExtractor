"""Contains the Keypoint class."""

from core.annotations import AnnotationId
from core.geometry import Point
from pydantic import BaseModel, ConfigDict, Field
from enum import StrEnum
from uuid import uuid4


class VisibilityStatus(StrEnum):
    """An enum with three variants based on a keypoint's visibility."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    MISSING = "missing"


class Keypoint(BaseModel):
    """A point with a classification.

    Attributes:
        point (Point):
            The point on the image for this keypoint.
        category (str):
            The keypoint's category. Cannot be empty.
        visibility (VisibilityStatus):
            The visibility of the keypoint.
        annotation_id (AnnotationId):
            The annotations unique identifier.
    """

    model_config = ConfigDict(frozen=True)

    point: Point
    category: str = Field(min_length=1)
    visibility: VisibilityStatus
    annotation_id: AnnotationId = Field(default_factory=uuid4)
