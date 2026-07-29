"""Contains the BoundingBoxWithKeypoints class."""

from core.annotations import AnnotationId, BoundingBox, Keypoint
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List
from typing_extensions import Self
from uuid import uuid4
import warnings


class BoundingBoxWithKeypoints(BaseModel):
    """A BoundingBox with one or more (ordered) keypoints associated with it.

    Attributes:
        bounding_box (BoundingBox):
            The bounding box.
        keypoints (List[Keypoint]):
            One or more keypoints associated with the box.
        annotation_id (AnnotationId):
            The annotations unique identifier.
    """

    model_config = ConfigDict(frozen=True)

    bounding_box: BoundingBox
    keypoints: List[Keypoint]
    annotation_id: AnnotationId = Field(default_factory=uuid4)

    @model_validator(mode="after")
    def warn_keypoints_not_contained_in_box(self) -> Self:
        """Warns the user if a keypoint(s) lie outside the bounding box.

        This is not impossible, but generally the keypoints are intended to lie within the box,
        making this a good opportunity to warn the user.
        """
        keypoints_not_contained_in_box: List[Keypoint] = list(
            filter(
                lambda kp: not self.bounding_box.rectangle.contains_point(kp.point),
                self.keypoints,
            )
        )
        for keypoint in keypoints_not_contained_in_box:
            warnings.warn(
                f"{keypoint} not contained in bounding box {self.bounding_box}",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def warn_no_keypoints(self) -> Self:
        """Warns the user if a BoundingBoxWithKeypoints is made with no keypoints."""
        if not self.keypoints:
            warnings.warn("BoundingBoxWithKeypoints has no keypoints.", UserWarning)
        return self
