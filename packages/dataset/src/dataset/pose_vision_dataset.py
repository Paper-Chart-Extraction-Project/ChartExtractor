"""Contains PoseVisionDataset, a child of VisionDataset."""

from functools import cached_property

from core.annotations import BoundingBoxWithKeypoints
from pydantic import Field

from .annotated_image import AnnotatedImage
from .vision_dataset import VisionDataset


class PoseVisionDataset(VisionDataset):
    """A vision dataset for bounding box detection with pose estimation."""

    annotated_images: list[AnnotatedImage[BoundingBoxWithKeypoints]] = Field(
        default_factory=list
    )

    @cached_property
    def names(self) -> list[str]:
        """A sorted list of the bounding box categories/names."""
        all_bbox_names: list[list[str]] = [
            [
                annotation.bounding_box.category
                for annotation in annotated_image.annotations
            ]
            for annotated_image in self.annotated_images
        ]
        sorted_unique_names: list[str] = sorted(
            {item for sublist in all_bbox_names for item in sublist}
        )

        return sorted_unique_names
