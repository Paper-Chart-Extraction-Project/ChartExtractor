"""Contains the AnnotatedImage class."""

from dataset.types import ImageId
from core.annotations import Annotation
from image import Image, ImageMetadata
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any
from uuid import uuid4
import warnings


class AnnotatedImage(BaseModel):
    """An image paired with annotations.

    Attributes:
        image (Image | ImageMetadata):
            The image that has annotations on it.
        annotations (list[Annotation]):
            The annotations on the image.
        image_id (ImageId):
            An identifier for the image. If left blank, sets itself to a UUID4, and emits a
            warning.
        image_metadata (dict[str, Any]):
            The image's metadata, if it has any.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image | ImageMetadata
    annotations: list[Annotation] = Field(default_factory=list)
    image_id: ImageId = Field(default=None, validate_default=True)
    image_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_id", mode="before")
    @classmethod
    def warn_if_no_image_id_supplied(cls, v: Any) -> ImageId:
        """Warns the user if no image id is supplied."""
        if v is None:
            warnings.warn(
                "No value supplied to AnnotatedImage for image_id, defaulting to a uuid4.",
                UserWarning,
                stacklevel=2,
            )
            return str(uuid4())
        return v
