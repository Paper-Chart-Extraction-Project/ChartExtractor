"""Tests the tiling module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from PIL import Image
import tiling


@pytest.fixture(scope="class", autouse=True)
def test_image(self):
    """Creates a PIL Image for testing purposes."""
    image = Image.new("L", (3, 3))
    image_data = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    image.putdata(image_data)
    return image


class TestValidateTileParameters:
    """Tests the validate_tile_image function."""

    pass
