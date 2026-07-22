"""A series of tests for BoundingBox."""

from core.geometry import BoundingBox
import pytest


class TestPointValidation:
    def test_top_left_higher_and_further_left_of_bottom_right(self):
        """Tests the normal construction of the BoundingBox with valid points."""
        BoundingBox.from_left_top_right_bottom(left=1.0, top=2.0, right=3.0, bottom=4.0)

    def test_top_left_lower_than_bottom_right(self):
        """Tests the model validator where the top left is lower than the bottom right."""
        with pytest.raises(ValueError):
            BoundingBox.from_left_top_right_bottom(
                left=1.0, top=4.0, right=3.0, bottom=2.0
            )

    def test_top_left_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is further right than the bottom right."""
        with pytest.raises(ValueError):
            BoundingBox.from_left_top_right_bottom(
                left=3.0, top=2.0, right=1.0, bottom=4.0
            )

    def test_top_left_lower_and_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is lower and further right than the bottom right."""
        with pytest.raises(ValueError):
            BoundingBox.from_left_top_right_bottom(
                left=3.0, top=4.0, right=1.0, bottom=2.0
            )


class TestDegeneracyWarning:
    def test_warning_left_equals_right(self):
        """Tests that a warning emits when the box's left equals its right."""
        with pytest.warns(UserWarning, match="left-right"):
            BoundingBox.from_left_top_right_bottom(
                left=5.0, top=4.0, right=5.0, bottom=10.0
            )

    def test_warning_top_equals_bottom(self):
        """Tests that a warning emits when the box's top equals its bottom."""
        with pytest.warns(UserWarning, match="top-bottom"):
            BoundingBox.from_left_top_right_bottom(
                left=4.0, top=5.0, right=10.0, bottom=5.0
            )

    def test_warning_top_equals_bottom_and_left_equals_right(self):
        """Tests that a warning emits when the box's left equals its right and top equals its bottom."""
        with pytest.warns(UserWarning, match="completely"):
            BoundingBox.from_left_top_right_bottom(
                left=4.0, top=5.0, right=4.0, bottom=5.0
            )


def test_from_center_xywh():
    """Tests the from_center_xywh constructor."""
    true_bounding_box: BoundingBox = BoundingBox.from_left_top_right_bottom(
        left=1.0, top=1.0, right=3.0, bottom=10.0
    )
    constructed_bounding_box: BoundingBox = BoundingBox.from_center_xywh(
        x_center=2.0, y_center=5.5, width=2.0, height=9.0
    )

    assert constructed_bounding_box == true_bounding_box
