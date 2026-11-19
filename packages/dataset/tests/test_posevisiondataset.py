"""A test module that tests the PoseVisionDataset class."""

import pytest
from core.annotations import (
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point, Rectangle
from image import ImageMetadata

from dataset import AnnotatedImage, PoseVisionDataset, Splits
from dataset.exceptions import DatasetError
from tests.helpers import generate_annotated_image


class TestKeypointNames:
    """A collection of tests for the keypoint_names property."""

    def test_keypoint_names_valid(self, dummy_image_metadata: ImageMetadata):
        """Tests the keypoint_names with valid inputs."""
        bbox_data: list[tuple[float, float, float, float, str]] = [
            (0.5, 0.5, 0.1, 0.1, "alpha"),
            (0.2, 0.8, 0.05, 0.07, "beta"),
            (0.7, 0.1, 0.02, 0.08, "alpha"),
        ]
        bboxes: list[BoundingBox] = [
            BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
            for (x, y, w, h, s) in bbox_data
        ]
        bboxes_w_keypoints: list[BoundingBoxWithKeypoints] = [
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[0],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.5, y=0.51),
                        category="a",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.5, y=0.51),
                        category="b",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[1],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.21, y=0.79),
                        category="c",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.21, y=0.79),
                        category="d",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[2],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.7, y=0.11),
                        category="a",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.7, y=0.11),
                        category="b",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
        ]

        ann_ims: list[AnnotatedImage] = [
            generate_annotated_image(
                "1",
                [bboxes_w_keypoints[0], bboxes_w_keypoints[1]],
                dummy_image_metadata,
            ),
            generate_annotated_image(
                "2",
                [bboxes_w_keypoints[1], bboxes_w_keypoints[2]],
                dummy_image_metadata,
            ),
            generate_annotated_image(
                "3",
                [bboxes_w_keypoints[0], bboxes_w_keypoints[2]],
                dummy_image_metadata,
            ),
        ]
        splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

        kpt_names: dict[str, list[str]] = PoseVisionDataset(
            annotated_images=ann_ims, splits=splits
        ).keypoint_names
        true_kpt_names: dict[str, list[str]] = {"alpha": ["a", "b"], "beta": ["c", "d"]}
        assert kpt_names == true_kpt_names

    def test_keypoint_names_invalid(self, dummy_image_metadata: ImageMetadata):
        """Tests the keypoint_names with valid inputs."""
        bbox_data: list[tuple[float, float, float, float, str]] = [
            (0.5, 0.5, 0.1, 0.1, "a"),
            (0.2, 0.8, 0.05, 0.07, "a"),
            (0.7, 0.1, 0.02, 0.08, "a"),
        ]
        bboxes: list[BoundingBox] = [
            BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
            for (x, y, w, h, s) in bbox_data
        ]
        bboxes_w_keypoints: list[BoundingBoxWithKeypoints] = [
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[0],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.5, y=0.51),
                        category="a",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.5, y=0.51),
                        category="b",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[1],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.21, y=0.79),
                        category="a",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.21, y=0.79),
                        category="b",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[2],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.7, y=0.11),
                        category="a",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                    Keypoint(
                        point=Point(x=0.7, y=0.11),
                        category="c",
                        visibility=VisibilityStatus.VISIBLE,
                    ),
                ],
            ),
        ]

        ann_ims: list[AnnotatedImage] = [
            generate_annotated_image(
                "1",
                [bboxes_w_keypoints[0], bboxes_w_keypoints[1]],
                dummy_image_metadata,
            ),
            generate_annotated_image(
                "2",
                [bboxes_w_keypoints[1], bboxes_w_keypoints[2]],
                dummy_image_metadata,
            ),
            generate_annotated_image(
                "3",
                [bboxes_w_keypoints[0], bboxes_w_keypoints[2]],
                dummy_image_metadata,
            ),
        ]
        splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

        with pytest.raises(DatasetError, match="  3"):
            _ = PoseVisionDataset(
                annotated_images=ann_ims, splits=splits
            ).keypoint_names
