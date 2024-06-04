"""Tests the tiling module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from PIL import Image
import tiling


@pytest.fixture()
def test_image():
    """Creates a PIL Image for testing purposes."""
    image = Image.new("L", (3, 3))
    image_data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image.putdata(image_data)
    return image


class TestValidateTileParameters:
    """Tests the validate_tile_image function."""

    def test_slice_width_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the slice width is less than or equal to 0."""
        with pytest.raises(ValueError, match="slice_width must be"):
            slice_width, slice_height = 0, 1
            horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_slice_width_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the slice width is greater than the image width."""
        with pytest.raises(ValueError, match="slice_width must be"):
            slice_width, slice_height = 4, 1
            horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_slice_height_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the slice height is less than or equal to 0."""
        pass

    def test_slice_height_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the slice height is greater than the image height."""
        pass

    def test_horizontal_overlap_ratio_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the horizontal overlap ratio is less than or equal to 0."""
        pass

    def test_horizontal_overlap_ratio_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the horizontal overlap ratio is greater than 1."""
        pass

    def test_vertical_overlap_ratio_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the vertical overlap ratio is less than or equal to 0."""
        pass

    def test_vertical_overlap_ratio_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the vertical overlap ratio is greater than 1."""
        pass
