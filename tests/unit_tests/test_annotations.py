""" """

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from annotations import BoundingBox, Keypoint, Point


class TestBoundingBox:
    """Tests the BoundingBox class."""

    def test_init(self):
        """Tests the init function with valid parameters."""
        BoundingBox("Test", 0, 0, 1, 1)

    def test_validate_box_values_left_greater_than_right(self):
        """Tests the validate_box_values classmethod with invalid parameters (left > right)."""
        with pytest.raises(ValueError, match="left side greater than its right side"):
            BoundingBox("Test", 1, 0, 0, 1)

    def test_validate_box_values_top_greater_than_bottom(self):
        """Tests the validate_box_values classmethod with invalid parameters (top > bottom)."""
        with pytest.raises(ValueError, match="top side greater than its bottom side"):
            BoundingBox("Test", 0, 1, 1, 0)

    def test_validate_box_values_left_eq_right(self):
        """Tests the validate_box_values classmethod with degenerate rectangle parameters (left == right)."""
        with pytest.warns(UserWarning, match="left side equals its right side"):
            BoundingBox("Test", 0, 0, 0, 1)

    def test_validate_box_values_top_eq_bottom(self):
        """Tests the validate_box_values classmethod with degenerate rectangle parameters (top == bottom)."""
        with pytest.warns(UserWarning, match="top side equals its bottom side"):
            BoundingBox("Test", 0, 0, 1, 0)

    def test_validate_box_values_left_eq_right_top_eq_bottom(self):
        """Tests the validate_box_values classmethod with degernate rectangle parameters (left == right == top == bottom)."""
        with pytest.warns(UserWarning, match="box's parameters are equal"):
            BoundingBox("Test", 0, 0, 0, 0)
