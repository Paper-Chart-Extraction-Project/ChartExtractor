"""A test module that tests the VisionDataset class."""

import pytest
from dataset import (
    AnnotatedImage,
    DatasetError,
    EmptyDatasetWarning,
    EmptySplitWarning,
    ImageId,
    OverlappingSplitsError,
    VisionDataset,
)
from image import ImageMetadata
from pydantic import ValidationError


@pytest.fixture
def image_metadata() -> ImageMetadata:
    """An ImageMetadata object for testing."""
    return ImageMetadata(path=None)


def generate_annotated_image(
    image_id: ImageId, image_metadata: ImageMetadata
) -> AnnotatedImage:
    """A function that generates 'dummy' annotated images.

    Args:
        image_id (ImageId):
            The id to assign the image.

    Returns:
        An AnnotatedImage with no real data and the supplied ImageId.
    """
    return AnnotatedImage(image=image_metadata, image_id=image_id)


class TestDatasetCreation:
    """A collection of tests for the construction of a VisionDataset."""

    def test_empty_dataset(self):
        """Tests the creation of a dataset with no images."""
        with pytest.warns(EmptyDatasetWarning, match="is empty"):
            VisionDataset(
                annotated_images=[],
                dataset_metadata={},
                splits={"train": [], "val": []},
            )

    def test_valid_dataset(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset with valid splits."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2"},
            "val": {"im_3"},
            "test": {"im_4"},
        }

        VisionDataset(
            annotated_images=annotated_images, dataset_metadata={}, splits=splits
        )

    def test_duplicate_image_ids(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset with valid splits."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in [0, 1, 2, 2, 3, 4, 5]
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2"},
            "val": {"im_3"},
            "test": {"im_4"},
        }

        with pytest.raises(DatasetError, match="with the same image id"):
            VisionDataset(
                annotated_images=annotated_images, dataset_metadata={}, splits=splits
            )

    def test_overlapping_splits(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset with splits that overlap."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2"},
            "val": {"im_3"},
            "test": {"im_4", "im_2"},
        }

        with pytest.raises(OverlappingSplitsError, match="in both sets"):
            VisionDataset(
                annotated_images=annotated_images, dataset_metadata={}, splits=splits
            )

    def test_image_missing_from_splits(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset where the splits don't fully cover the dataset."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_1", "im_2"},
            "val": {"im_3"},
            "test": {"im_4"},
        }

        with pytest.raises(DatasetError, match="do not span"):
            VisionDataset(
                annotated_images=annotated_images, dataset_metadata={}, splits=splits
            )

    def test_necessary_split_missing(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset when a necessary split (val) is missing."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "test": {"im_4"},
        }

        with pytest.raises(ValidationError, match="Field required"):
            VisionDataset(
                annotated_images=annotated_images, dataset_metadata={}, splits=splits
            )

    def test_unnecessary_split_missing(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset when an unecessary split (test) is missing."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }

        VisionDataset(
            annotated_images=annotated_images, dataset_metadata={}, splits=splits
        )

    def test_split_is_empty(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset when a split has no image ids in it."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
            "test": set(),
        }

        with pytest.warns(EmptySplitWarning, match="exists but is empty"):
            VisionDataset(
                annotated_images=annotated_images, dataset_metadata={}, splits=splits
            )
