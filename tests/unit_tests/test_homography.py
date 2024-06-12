"""Tests the homography module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from PIL import Image
from image_registration.homography import homography_transform


@pytest.fixture()
def test_pil_image() -> Image.Image:
    """Creates a PIL Image for testing purposes."""
    image = Image.new("RGB", (3, 3))
    return image


class TestHomographyTransform:
    """Tests the homography_transform function."""

    def test_not_enough_points(self, test_pil_image):
        """Tests the homography_tranform function when there are not enough points."""
        with pytest.raises(ValueError, match="Must have 4"):
            homography_transform(
                test_pil_image, [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (5, 6)]
            )

    def test_unequal_number_of_points(self):
        """Tests the homography_transform function when there are an unequal number of src and dst points."""
        with pytest.raises(ValueError, match="Source and destination points"):
            homography_transform(
                test_pil_image, [(0, 1), (2, 3)], [(1, 2), (3, 4), (5, 6)]
            )
