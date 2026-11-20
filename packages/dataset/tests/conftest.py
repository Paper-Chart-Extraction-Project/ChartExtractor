"""A module with test fixtures shared between files."""

import pytest
from core.annotations import BoundingBox
from core.geometry import Rectangle
from image import ImageMetadata

from dataset import AnnotatedImage, DetectVisionDataset, Splits
from tests.helpers import generate_annotated_image


@pytest.fixture
def dummy_image_metadata() -> ImageMetadata:
    """An ImageMetadata object for testing."""
    return ImageMetadata(path=None)


@pytest.fixture
def detect_vision_dataset(dummy_image_metadata: ImageMetadata) -> DetectVisionDataset:
    """A detect vision dataset for testing."""
    bbox_data: list[tuple[float, float, float, float, str]] = [
        (0.5, 0.5, 0.1, 0.1, "a"),
        (0.2, 0.8, 0.05, 0.07, "b"),
        (0.7, 0.1, 0.02, 0.08, "a"),
    ]
    bboxes: list[BoundingBox] = [
        BoundingBox(rectangle=Rectangle.from_center_xywh(x, y, w, h), category=s)
        for (x, y, w, h, s) in bbox_data
    ]

    ann_ims: list[AnnotatedImage[BoundingBox]] = [
        generate_annotated_image("1", [bboxes[0], bboxes[1]], dummy_image_metadata),
        generate_annotated_image("2", [bboxes[1], bboxes[2]], dummy_image_metadata),
        generate_annotated_image("3", [bboxes[0], bboxes[2]], dummy_image_metadata),
    ]
    splits: Splits = {"train": {"1", "2"}, "val": {"3"}}

    return DetectVisionDataset(annotated_images=ann_ims, splits=splits)
