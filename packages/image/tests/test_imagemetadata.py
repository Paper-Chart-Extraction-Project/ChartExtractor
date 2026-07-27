"""A series of tests for the ImageMetadata class."""

from pathlib import Path

import pytest
from conftest import TEST_DATA_DIR

from image import Image, ImageMetadata


class TestPathIsValid:
    """A group of tests for the check_path_is_valid field validator."""

    def test_valid_filepath(self, three_by_three_rgb_test_path: Path):
        """Tests if no error is thrown for a valid filepath."""
        ImageMetadata(path=three_by_three_rgb_test_path)

    def test_non_existant_filepath(self):
        """Tests if an error is thrown when validating a non-existant filepath."""
        non_existant_filepath: Path = TEST_DATA_DIR / "non_existant_image.jpg"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ImageMetadata(path=non_existant_filepath)

    def test_filepath_is_a_directory(self):
        """Tests if an error is thrown when validating a filepath to a directory."""
        with pytest.raises(IsADirectoryError, match="is a directory"):
            ImageMetadata(path=TEST_DATA_DIR)

    def test_none(self):
        """Tests if the model validator passes 'None' through."""
        ImageMetadata(path=None)


class TestSize:
    """A group of tests for the size property."""

    def test_size_is_none_when_path_is_none(self):
        """Tests that the size property is None when path=None."""

        assert ImageMetadata(path=None).size == None

    def test_three_by_three_rgb_image(self, three_by_three_rgb_test_path: Path):
        """Tests the size property with an image that is 3x3 rgb."""
        metadata: ImageMetadata = ImageMetadata(path=three_by_three_rgb_test_path)

        assert metadata.size == (3, 3, 3)

    def test_four_by_three_black_image(self, four_by_three_black_test_path: Path):
        """Tests the size property with a 4x3 image with no exif data."""
        metadata: ImageMetadata = ImageMetadata(path=four_by_three_black_test_path)

        assert metadata.size == (3, 4, 3)

    def test_four_by_three_exif_rotated_black_image(
        self, four_by_three_exif_rotated_black_test_path: Path
    ):
        """Tests the size property with a 4x3 image with no exif data."""
        metadata: ImageMetadata = ImageMetadata(
            path=four_by_three_exif_rotated_black_test_path
        )

        assert metadata.size == (4, 3, 3)

    def test_four_by_three_exif_rotated_black_image_skip_exif_transpose(
        self, four_by_three_exif_rotated_black_test_path: Path
    ):
        """Tests the size property with a 4x3 image that has exif data, but skips applying it."""
        metadata: ImageMetadata = ImageMetadata(
            path=four_by_three_exif_rotated_black_test_path, skip_exif_transpose=True
        )

        assert metadata.size == (3, 4, 3)

    def test_three_by_three_greyscale_image(
        self, three_by_three_greyscale_test_path: Path
    ):
        """Tests the size property with a single channel image."""
        metadata: ImageMetadata = ImageMetadata(path=three_by_three_greyscale_test_path)

        assert metadata.size == (3, 3, 1)


def test_load(three_by_three_greyscale_test_path: Path):
    """Tests that the load() method produces an image with the metadata attached."""
    metadata: ImageMetadata = ImageMetadata(path=three_by_three_greyscale_test_path)
    im: Image = metadata.load()

    assert im._metadata == metadata
