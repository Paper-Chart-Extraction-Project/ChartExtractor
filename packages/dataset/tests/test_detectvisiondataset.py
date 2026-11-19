"""A test module that tests the DetectVisionDataset class."""

import pytest
from core.annotations import (
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point, Rectangle
from image import ImageMetadata
from pydantic import ValidationError

from dataset import DetectVisionDataset, Splits
from dataset.annotated_image import AnnotatedImage

from .helpers import generate_annotated_image


class TestDetectDatasetCreation:
    """A collection of tests for the construction of a DetectVisionDataset."""

    def test_all_bounding_boxes(self, dummy_image_metadata: ImageMetadata):
        """Tests the construction with annotated images that have only bounding boxes."""
        bbox_data: list[tuple[float, float, float, float, str]] = [
            (0.5, 0.5, 0.1, 0.1, "a"),
            (0.2, 0.8, 0.05, 0.07, "b"),
            (0.7, 0.1, 0.02, 0.08, "a"),
        ]
        bboxes: list[BoundingBox] = [
            BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
            for (x, y, w, h, s) in bbox_data
        ]

        ann_ims: list[AnnotatedImage] = [
            generate_annotated_image("1", [bboxes[0], bboxes[1]], dummy_image_metadata),
            generate_annotated_image("2", [bboxes[1], bboxes[2]], dummy_image_metadata),
            generate_annotated_image("3", [bboxes[0], bboxes[2]], dummy_image_metadata),
        ]
        splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

        DetectVisionDataset(annotated_images=ann_ims, splits=splits)

    def test_not_all_bounding_boxes(self, dummy_image_metadata: ImageMetadata):
        """Tests the construction with annotated images that have some non-bounding box annotations."""
        bbox_data: list[tuple[float, float, float, float, str]] = [
            (0.5, 0.5, 0.1, 0.1, "a"),
            (0.2, 0.8, 0.05, 0.07, "b"),
            (0.7, 0.1, 0.02, 0.08, "a"),
        ]
        bboxes: list[BoundingBox] = [
            BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
            for (x, y, w, h, s) in bbox_data
        ]
        bbox_w_keypoints: BoundingBoxWithKeypoints = BoundingBoxWithKeypoints(
            bounding_box=bboxes[0],
            keypoints=[
                Keypoint(
                    point=Point(x=0.5, y=0.51),
                    category="d",
                    visibility=VisibilityStatus.VISIBLE,
                )
            ],
        )

        ann_ims: list[AnnotatedImage] = [
            generate_annotated_image("1", [bboxes[0], bboxes[1]], dummy_image_metadata),
            generate_annotated_image("2", [bboxes[1], bboxes[2]], dummy_image_metadata),
            generate_annotated_image(
                "3",
                [bboxes[0], bboxes[2], bbox_w_keypoints],  # pyright: ignore
                dummy_image_metadata,
            ),
        ]
        splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

        with pytest.raises(ValidationError):
            DetectVisionDataset(annotated_images=ann_ims, splits=splits)

    def test_no_bounding_boxes(self, dummy_image_metadata: ImageMetadata):
        """Tests the construction with annotated images that have no bounding box annotations."""
        bbox_data: list[tuple[float, float, float, float, str]] = [
            (0.5, 0.5, 0.1, 0.1, "a"),
            (0.2, 0.8, 0.05, 0.07, "b"),
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
                        category="d",
                        visibility=VisibilityStatus.VISIBLE,
                    )
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[1],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.21, y=0.79),
                        category="d",
                        visibility=VisibilityStatus.VISIBLE,
                    )
                ],
            ),
            BoundingBoxWithKeypoints(
                bounding_box=bboxes[2],
                keypoints=[
                    Keypoint(
                        point=Point(x=0.7, y=0.11),
                        category="d",
                        visibility=VisibilityStatus.VISIBLE,
                    )
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

        with pytest.raises(ValidationError):
            DetectVisionDataset(annotated_images=ann_ims, splits=splits)


def test_names(dummy_image_metadata: ImageMetadata):
    """Tests the names property."""
    bbox_data: list[tuple[float, float, float, float, str]] = [
        (0.5, 0.5, 0.1, 0.1, "b"),
        (0.2, 0.8, 0.05, 0.07, "b"),
        (0.7, 0.1, 0.02, 0.08, "a"),
    ]
    bboxes: list[BoundingBox] = [
        BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
        for (x, y, w, h, s) in bbox_data
    ]

    ann_ims: list[AnnotatedImage] = [
        generate_annotated_image("1", [bboxes[0], bboxes[1]], dummy_image_metadata),
        generate_annotated_image("2", [bboxes[1], bboxes[2]], dummy_image_metadata),
        generate_annotated_image("3", [bboxes[0], bboxes[2]], dummy_image_metadata),
    ]
    splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

    dvd: DetectVisionDataset = DetectVisionDataset(
        annotated_images=ann_ims, splits=splits
    )

    assert dvd.names == ["a", "b"]
