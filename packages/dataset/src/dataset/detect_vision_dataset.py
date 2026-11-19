"""Contains the DetectVisionDataset class, a child of VisionDataset."""

from core.annotations import BoundingBox
from pydantic import field_validator

from .annotated_image import AnnotatedImage
from .exceptions import DatasetError
from .vision_dataset import VisionDataset


class DetectVisionDataset(VisionDataset):
    """A vision dataset for detecting bounding boxes."""

    @field_validator("annotated_images")
    @classmethod
    def check_all_annotations_are_bounding_boxes(
        cls, annotated_images: list[AnnotatedImage]
    ):
        """Checks to make sure that all the annotations in the vision dataset are BoundingBoxes."""
        non_bounding_box_annotations: list[tuple[str, str]] = []
        for annotated_image in annotated_images:
            for annotation in annotated_image.annotations:
                if not isinstance(annotation, BoundingBox):
                    non_bounding_box_annotations.append(
                        (str(type(annotation)), annotated_image.image_id)
                    )
        if non_bounding_box_annotations:
            err_msg: list[str] = ["All annotations in a DetectVisionDataset must be of type BoundingBox."]
            for non_bb_type, image_id in non_bounding_box_annotations:
                err_msg.append(f"Found: {non_bb_type} on image {image_id}.")
            raise DatasetError("\n".join(err_msg))

        return annotated_images
