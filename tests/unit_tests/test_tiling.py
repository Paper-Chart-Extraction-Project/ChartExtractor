"""Tests the tiling module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from typing import List
from PIL import Image, ImageChops
import tiling
from annotations import BoundingBox


@pytest.fixture()
def test_image() -> Image.Image:
    """Creates a PIL Image for testing purposes."""
    image = Image.new("L", (3, 3))
    image_data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image.putdata(image_data)
    return image


@pytest.fixture()
def test_annotations() -> List[BoundingBox]:
    """Creates a short list of annotations for testing."""
    return [
        BoundingBox("Test", 0, 0, 1, 1),
        BoundingBox("Test", 2, 2, 3, 3),
        BoundingBox("Test", 3, 3, 4, 4),
    ]


def test_tile_image(test_image):
    """Function that tests tile_image"""

    def image_from_list(l, size):
        image = Image.new("L", (size, size))
        image.putdata(l)
        return image

    slice_width, slice_height = 2, 2
    horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
    created_tiles = tiling.tile_image(
        test_image,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    true_tiles = [
        [
            image_from_list([0, 1, 3, 4], 2),
            image_from_list([1, 2, 4, 5], 2),
            image_from_list([2, 0, 5, 0], 2),
        ],
        [
            image_from_list([3, 4, 6, 7], 2),
            image_from_list([4, 5, 7, 8], 2),
            image_from_list([5, 0, 8, 0], 2),
        ],
        [
            image_from_list([6, 7, 0, 0], 2),
            image_from_list([7, 8, 0, 0], 2),
            image_from_list([8, 0, 0, 0], 2),
        ],
    ]
    assert len(true_tiles) == len(created_tiles)
    assert len(true_tiles[0]) == len(created_tiles[0])
    for ix, im_list in enumerate(created_tiles):
        for jx, im in enumerate(im_list):
            diff = ImageChops.difference(im, true_tiles[ix][jx])
            assert diff.getbbox() is None


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
    """Function that tests generate_tile_coordinates."""
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


def test_tile_annotations(test_annotations):
    """Function that tests tile_annotations."""
    image_width, image_height = 4, 4
    slice_width, slice_height = 3, 3
    horizontal_overlap_ratio, vertical_overlap_ratio = 0.5, 0.5
    created_tiled_annotations = tiling.tile_annotations(
        test_annotations,
        image_width,
        image_height,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    true_tiled_annotations = [
        [[test_annotations[0], test_annotations[1]], [test_annotations[1]]],
        [[test_annotations[1]], [test_annotations[1], test_annotations[2]]],
    ]
    assert created_tiled_annotations == true_tiled_annotations


def test_get_annotations_in_tile(test_annotations):
    """Function that tests get_annotations_in_tile."""
    created_annotations_in_tile = tiling.get_annotations_in_tile(
        test_annotations, [0, 0, 3, 3]
    )
    true_annotations_in_tile = [test_annotations[0], test_annotations[1]]
    assert created_annotations_in_tile == true_annotations_in_tile
