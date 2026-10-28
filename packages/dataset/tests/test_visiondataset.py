"""A test module that tests the VisionDataset class."""

import pytest
from dataset import (
    AnnotatedImage,
    DatasetError,
    EmptyDatasetWarning,
    EmptyEvaluationSetWarning,
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
                splits={"train": set(), "val": set()},
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
            annotated_images=annotated_images,
            dataset_metadata={},
            splits=splits,
            evaluation_sets={},
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
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
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
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
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
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
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
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
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
            annotated_images=annotated_images,
            dataset_metadata={},
            splits=splits,
            evaluation_sets={},
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
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
            )

    def test_extra_key_in_split_dictionary(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset when there is an extra entry in the splits dictionary."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
            "test": set(),
            "test_2": set(),
        }

        with pytest.raises(DatasetError, match="Extra keys"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets={},
            )

    def test_eval_sets_with_valid_names(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset with valid evaluation sets."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {
            "even_images": {"im_0", "im_2", "im_4"}
        }

        VisionDataset(
            annotated_images=annotated_images,
            dataset_metadata={},
            splits=splits,
            evaluation_sets=evaluation_sets,
        )

    def test_eval_sets_with_invalid_names(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset with invalid evaluation sets."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {"train": {"im_0", "im_2", "im_4"}}

        with pytest.raises(DatasetError, match=": train"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )

        evaluation_sets: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_2", "im_4"},
            "val": set(),
        }

        with pytest.raises(DatasetError, match="may not be named"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )

    def test_eval_set_with_blank_name(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset with an valid evaluation set that is named ''."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {"": {"im_0", "im_2", "im_4"}}

        with pytest.raises(DatasetError, match="empty string"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )

    def test_splits_with_image_ids_not_in_dataset(self, image_metadata: ImageMetadata):
        """Tests the creation of a dataset with splits that include an image not in the dataset."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
            "test": {"im_6"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {}

        with pytest.raises(DatasetError, match="that don't exist in the dataset"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )

    def test_eval_set_with_image_ids_not_in_dataset(
        self, image_metadata: ImageMetadata
    ):
        """Test the creation of a dataset with an evaluation set that includes an image not in the dataset."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {
            "even_images": {"im_0", "im_2", "im_4"},
            "negative_images": {"im_-1"},
        }

        with pytest.raises(DatasetError, match="that don't exist in the dataset"):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )

    def test_eval_sets_that_overlap(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset with valid evaluation sets that share image ids."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {
            "even_images": {"im_0", "im_2", "im_4"},
            "positive_images_mod_2": {"im_2", "im_4"},
        }

        VisionDataset(
            annotated_images=annotated_images,
            dataset_metadata={},
            splits=splits,
            evaluation_sets=evaluation_sets,
        )

    def test_empty_eval_set(self, image_metadata: ImageMetadata):
        """Test the creation of a dataset with an evaluation set that is empty."""
        annotated_images: list[AnnotatedImage] = [
            generate_annotated_image(f"im_{im_id}", image_metadata)
            for im_id in range(5)
        ]
        splits: dict[str, set[ImageId]] = {
            "train": {"im_0", "im_1", "im_2", "im_3"},
            "val": {"im_4"},
        }
        evaluation_sets: dict[str, set[ImageId]] = {
            "even_images": {"im_0", "im_2", "im_4"},
            "positive_images_mod_2": {"im_2", "im_4"},
            "images_mod_5": set(),
        }

        with pytest.warns(EmptyEvaluationSetWarning, match="exists but is empty."):
            VisionDataset(
                annotated_images=annotated_images,
                dataset_metadata={},
                splits=splits,
                evaluation_sets=evaluation_sets,
            )
