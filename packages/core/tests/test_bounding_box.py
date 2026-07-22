"""A series of tests for BoundingBox."""

from core.geometry import BoundingBox, Point
import pytest
import warnings


class TestPointValidation:
    def test_top_left_higher_and_further_left_of_bottom_right(self):
        """Tests the normal construction of the BoundingBox with valid points."""
        BoundingBox.from_left_top_right_bottom(left=1.0, top=2.0, right=3.0, bottom=4.0)

    def test_top_left_lower_than_bottom_right(self):
        """Tests the model validator where the top left is lower than the bottom right."""
        with pytest.raises(ValueError, match="is lower than"):
            BoundingBox.from_left_top_right_bottom(
                left=1.0, top=4.0, right=3.0, bottom=2.0
            )

    def test_top_left_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is further right than the bottom right."""
        with pytest.raises(ValueError, match="is further right than"):
            BoundingBox.from_left_top_right_bottom(
                left=3.0, top=2.0, right=1.0, bottom=4.0
            )

    def test_top_left_lower_and_further_right_than_bottom_right(self):
        """Tests the model validator where the top left is lower and further right than the bottom right."""
        with pytest.raises(ValueError, match="is lower and further right than"):
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

    def test_nearly_degenerate(self):
        """Tests that boxes very close to degenerate don't falsely warn."""
        # This should NOT warn (difference is larger than the hard-coded rel_tol=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            BoundingBox.from_left_top_right_bottom(
                left=1.0, top=2.0, right=1.0 + 1e-8, bottom=10.0
            )


class TestNegativeWidthHeightFromTopLeftXYWH:
    """A group of tests that ensures the width/height validation in the from_top_left_xywh constructor."""

    def test_from_top_left_xywh_negative_width(self):
        """Tests that negative width raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            BoundingBox.from_top_left_xywh(left=0, top=0, width=-5, height=10)

    def test_from_top_left_xywh_negative_height(self):
        """Tests that negative height raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            BoundingBox.from_top_left_xywh(left=0, top=0, width=5, height=-10)

    def test_from_top_left_xywh_both_negative(self):
        """Tests that both negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            BoundingBox.from_top_left_xywh(left=0, top=0, width=-5, height=-10)


class TestConstructorRoundTrips:
    """A series of sanity-check tests that ensure going to and from BoundingBox 'encodings' works."""

    ORIGINAL_BBOX: BoundingBox = BoundingBox.from_left_top_right_bottom(
        left=2.0, top=3.0, right=8.0, bottom=11.0
    )

    def test_to_center_xywh_round_trip(self):
        """Tests that converting to center+width/height format and back works."""
        x_center, y_center, width, height = self.ORIGINAL_BBOX.to_center_xywh()
        reconstructed = BoundingBox.from_center_xywh(x_center, y_center, width, height)

        assert reconstructed == self.ORIGINAL_BBOX

    def test_to_top_left_xywh_round_trip(self):
        """Tests that converting to top-left format and back works."""
        left, top, width, height = self.ORIGINAL_BBOX.to_top_left_xywh()
        reconstructed = BoundingBox.from_top_left_xywh(left, top, width, height)

        assert reconstructed == self.ORIGINAL_BBOX

    def test_to_left_top_right_bottom_round_trip(self):
        """Tests that converting to ltrb format and back works."""
        left, top, right, bottom = self.ORIGINAL_BBOX.to_left_top_right_bottom()
        reconstructed = BoundingBox.from_left_top_right_bottom(left, top, right, bottom)

        assert reconstructed == self.ORIGINAL_BBOX


def test_from_center_xywh():
    """Tests the from_center_xywh constructor."""
    true_bounding_box: BoundingBox = BoundingBox.from_left_top_right_bottom(
        left=1.0, top=1.0, right=3.0, bottom=10.0
    )
    constructed_bounding_box: BoundingBox = BoundingBox.from_center_xywh(
        x_center=2.0, y_center=5.5, width=2.0, height=9.0
    )

    assert constructed_bounding_box == true_bounding_box


def test_computed_properties():
    """Tests that all computed properties return expected values."""
    bbox = BoundingBox.from_left_top_right_bottom(
        left=2.0, top=3.0, right=8.0, bottom=11.0
    )

    assert bbox.left == 2.0
    assert bbox.top == 3.0
    assert bbox.right == 8.0
    assert bbox.bottom == 11.0
    assert bbox.width == 6.0
    assert bbox.height == 8.0
    assert bbox.area == 48.0
    assert bbox.center == Point(x=5.0, y=7.0)
    assert bbox.top_right == Point(x=8.0, y=3.0)
    assert bbox.bottom_left == Point(x=2.0, y=11.0)


def test_all_constructors_produce_same_result():
    """Tests that all constructors create the same BoundingBox.

    Tests that the constructors all create the same BoundingBox when supplied with parameters
    that *ought* to create the same BoundingBox internally.
    """
    # Direct construction
    bbox1 = BoundingBox.from_left_top_right_bottom(
        left=2.0, top=3.0, right=8.0, bottom=11.0
    )

    # From top-left
    bbox2 = BoundingBox.from_top_left_xywh(left=2.0, top=3.0, width=6.0, height=8.0)

    # From center
    bbox3 = BoundingBox.from_center_xywh(
        x_center=5.0, y_center=7.0, width=6.0, height=8.0
    )

    assert bbox1 == bbox2 == bbox3
