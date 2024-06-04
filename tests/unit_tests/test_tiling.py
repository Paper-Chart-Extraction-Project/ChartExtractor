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
    """Class that organizes test functions for validate_tile_parameters."""

    def test_slice_width_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the slice width is less than or equal to 0."""
        slice_width, slice_height = 0, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
        with pytest.raises(ValueError, match="slice_width must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_slice_width_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the slice width is greater than the image width."""
        slice_width, slice_height = 4, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
        with pytest.raises(ValueError, match="slice_width must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_slice_height_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the slice height is less than or equal to 0."""
        slice_width, slice_height = 1, 0
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
        with pytest.raises(ValueError, match="slice_height must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_slice_height_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the slice height is greater than the image height."""
        slice_width, slice_height = 1, 4
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
        with pytest.raises(ValueError, match="slice_height must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_horizontal_overlap_ratio_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the horizontal overlap ratio is less than or equal to 0."""
        slice_width, slice_height = 1, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.0, 0.5
        with pytest.raises(ValueError, match="horizontal_overlap_ratio must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_horizontal_overlap_ratio_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the horizontal overlap ratio is greater than 1."""
        slice_width, slice_height = 1, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 1.1, 0.5
        with pytest.raises(ValueError, match="horizontal_overlap_ratio must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_vertical_overlap_ratio_too_small(self, test_image):
        """Tests the validate_tile_parameters function when the vertical overlap ratio is less than or equal to 0."""
        slice_width, slice_height = 1, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.0
        with pytest.raises(ValueError, match="vertical_overlap_ratio must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )

    def test_vertical_overlap_ratio_too_large(self, test_image):
        """Tests the validate_tile_parameters function when the vertical overlap ratio is greater than 1."""
        slice_width, slice_height = 1, 1
        horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 1.1
        with pytest.raises(ValueError, match="vertical_overlap_ratio must be"):
            tiling.validate_tile_parameters(
                test_image,
                slice_width,
                slice_height,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )


def test_generate_tile_coordinates():
    """Class that organizes test functions for generate_tile_coordinates."""

    image_width, image_height = 8, 8
    slice_width, slice_height = 4, 4
    horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5

    created_tile_coordinates = tiling.generate_tile_coordinates(
        image_width,
        image_height,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    true_tile_coordinates = [
        [(0, 0, 4, 4), (2, 0, 6, 4), (4, 0, 8, 4), (6, 0, 10, 4)],
        [(0, 2, 4, 6), (2, 2, 6, 6), (4, 2, 8, 6), (6, 2, 10, 6)],
        [(0, 4, 4, 8), (2, 4, 6, 8), (4, 4, 8, 8), (6, 4, 10, 8)],
        [(0, 6, 4, 10), (2, 6, 6, 10), (4, 6, 8, 10), (6, 6, 10, 10)],
    ]
    assert created_tile_coordinates == true_tile_coordinates
