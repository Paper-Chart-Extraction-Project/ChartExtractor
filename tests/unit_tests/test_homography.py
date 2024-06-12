"""Tests the homography module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

from image_registration import homography


class TestHomographyTransform:
    """Tests the homography_transform function."""

    def test_not_enough_points():
        """Tests the homography_tranform function when there are not enough points."""
        pass

    def test_unequal_number_of_points():
        """Tests the homography_transform function when there are an unequal number of src and dst points."""
        pass
