"""A series of tests for BoundingBox."""

from core.geometry import BoundingBox
from core.geometry import Point
import pytest


class TestPointValidation:
    def test_top_left_higher_and_further_left_of_bottom_right(self):
        """Tests the normal construction of the BoundingBox with valid points."""
        top_left: Point = Point(x=1.0, y=2.0)
        bottom_right: Point = Point(x=3.0, y=4.0)
        BoundingBox(top_left=top_left, bottom_right=bottom_right)

    def test_top_left_lower_than_bottom_right(self):
        """Tests the model validator where the top left is lower than the bottom right."""
        top_left: Point = Point(x=1.0, y=4.0)
        bottom_right: Point = Point(x=3.0, y=2.0)
        with pytest.raises(ValueError):
            BoundingBox(top_left=top_left, bottom_right=bottom_right)

    def test_top_left_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is further right than the bottom right."""
        top_left: Point = Point(x=3.0, y=2.0)
        bottom_right: Point = Point(x=1.0, y=4.0)
        with pytest.raises(ValueError):
            BoundingBox(top_left=top_left, bottom_right=bottom_right)

    def test_top_left_lower_and_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is lower and further right than the bottom right."""
        top_left: Point = Point(x=3.0, y=4.0)
        bottom_right: Point = Point(x=1.0, y=2.0)
        with pytest.raises(ValueError):
            BoundingBox(top_left=top_left, bottom_right=bottom_right)


def test_from_center_xywh():
    """Tests the from_center_xywh constructor."""
    true_bounding_box: BoundingBox = BoundingBox(
        top_left=Point(x=1.0, y=1.0), bottom_right=Point(x=3.0, y=10.0)
    )
    center_x: float = 2.0
    center_y: float = 5.5
    width: float = 2.0
    height: float = 9.0

    constructed_bounding_box: BoundingBox = BoundingBox.from_center_xywh(
        center_x, center_y, width, height
    )

    assert constructed_bounding_box == true_bounding_box
