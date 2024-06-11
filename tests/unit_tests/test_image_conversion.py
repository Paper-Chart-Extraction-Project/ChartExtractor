"""Tests for the image_conversion module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from PIL import Image, ImageChops
import cv2
import numpy as np
from utilities import image_conversion


@pytest.fixture()
def test_pil_image() -> Image.Image:
    """Creates a PIL Image for testing purposes."""
    image = Image.new("RGB", (3, 3))
    return image


@pytest.fixture()
def test_cv2_image() -> np.ndarray:
    """Creates a cv2 image for testing purposes."""
    return np.zeros((3, 3, 3), np.uint8)


class TestPilToCv2:
    """Tests the pil_to_cv2 function."""

    def test_typical_inputs(self, test_pil_image):
        """Tests the pil_to_cv2 function with typical inputs."""
        true_cv2_image = np.array([[(0, 0, 0) for i in range(3)] for j in range(3)])
        created_cv2_image = image_conversion.pil_to_cv2(test_pil_image)
        assert np.array_equal(true_cv2_image, created_cv2_image)

    def test_bad_mode(self):
        """Tests the pil_to_cv2 functions ability to throw an error when the image's mode is not allowed."""
        image = Image.new("L", (3, 3))
        image_data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        image.putdata(image_data)
        with pytest.raises(ValueError):
            image_conversion.pil_to_cv2(image)


class TestCv2ToPil:
    """Tests the cv2_to_pil function."""

    def test_typical_inputs(self, test_cv2_image):
        """Tests the cv2_to_pil function with typical inputs."""
        true_pil_image = Image.new("RGB", (3, 3))
        created_pil_image = image_conversion.cv2_to_pil(test_cv2_image)
        diff = ImageChops.difference(true_pil_image, created_pil_image)
        assert diff.getbbox() is None
