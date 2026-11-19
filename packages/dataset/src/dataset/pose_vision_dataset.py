"""Contains PoseVisionDataset, a child of VisionDataset."""

from collections import Counter
from functools import cached_property
from typing import Self

from core.annotations import BoundingBoxWithKeypoints
from pydantic import Field, model_validator

from dataset.annotated_image import ImageId
from dataset.exceptions import DatasetError

from .annotated_image import AnnotatedImage
from .vision_dataset import VisionDataset


class PoseVisionDataset(VisionDataset):
    """A vision dataset for bounding box detection with pose estimation."""

    annotated_images: list[AnnotatedImage[BoundingBoxWithKeypoints]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_bounding_boxes_of_the_same_class_have_identical_keypoint_names(
        self,
    ) -> Self:
        """Ensures that all the bounding boxes of a particular category have the same keypoints.

        Goes category by category and checks that all the bounding boxes in that category have
        the same number and kind of keypoints. If not, prints the exact image ids that are likely
        to be in error.
        """
        for bbox_name in self.names:
            # collect all the keypoint names into one list paired to their image ids.
            bbox_keypoint_names: list[tuple[ImageId, list[str]]] = []
            for annotated_image in self.annotated_images:
                for annotation in annotated_image.annotations:
                    if annotation.bounding_box.category != bbox_name:
                        continue
                    bbox_keypoint_names.append(
                        (
                            annotated_image.image_id,
                            [kp.category for kp in annotation.keypoints],
                        )
                    )

            # Check to make sure (1) all the names are the same and (2) there are keypoints
            # with the same name.
            # If not, collect all the image ids that have keypoint tuples that aren't the most
            # common and print their image ids.
            all_keypoints_are_the_same = (
                not bbox_keypoint_names  # if the list is empty.
                or all(
                    bbox_kpts[1] == bbox_keypoint_names[0][1]
                    for bbox_kpts in bbox_keypoint_names
                )
            )
            if not all_keypoints_are_the_same:
                keypoint_lists_without_image_ids: list[list[str]] = [
                    l[1] for l in bbox_keypoint_names
                ]
                # Creates a counter that maps determines the most commonly seen list of keypoint
                # names. This is considered the correct one, and any other list that deviates
                # is printed as likely being in error.
                unique_keypoint_arrangements_w_counts: Counter = Counter(
                    map(tuple, keypoint_lists_without_image_ids)
                )
                err_msg_lines: list[str] = [
                    f"All keypoints for bounding box category {bbox_name} are not the same.",
                    "Image Ids that are likely in error:",
                ]
                for (
                    keypoint_arrangement,
                    _,
                ) in unique_keypoint_arrangements_w_counts.most_common()[1:]:
                    for im_id, bbox_kpts in bbox_keypoint_names:
                        if tuple(bbox_kpts) == keypoint_arrangement:
                            err_msg_lines.append(" " * 2 + f"{im_id}")
                raise DatasetError("\n".join(err_msg_lines))

        return self

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

    @cached_property
    def keypoint_names(self) -> dict[str, list[str]]:
        """A sorted list of the name of keypoints for each bounding box."""
        kpt_names: dict[str, list[str]] = {}
        for bbox_name in self.names:
            bbox_keypoint_names: list[tuple[ImageId, list[str]]] = []
            for annotated_image in self.annotated_images:
                for annotation in annotated_image.annotations:
                    if annotation.bounding_box.category != bbox_name:
                        continue
                    bbox_keypoint_names.append(
                        (
                            annotated_image.image_id,
                            [kp.category for kp in annotation.keypoints],
                        )
                    )

            kpt_names[bbox_name] = sorted(set(bbox_keypoint_names[0][1]))

        return kpt_names
