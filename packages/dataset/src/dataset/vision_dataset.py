"""Contains the VisionDataset class."""

from dataset import AnnotatedImage
from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class VisionDataset(BaseModel):
    """A computer vision dataset that pairs images with annotations.

    Attributes:
        annotated_images (list[AnnotatedImage]):
            The images with annotations that form the dataset.
        dataset_metadata (dict[str, Any]):
            The datasets metadata, if it has any.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    annotated_images: list[AnnotatedImage] = Field(default_factory=list)
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
