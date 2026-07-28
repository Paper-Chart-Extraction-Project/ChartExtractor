"""A series of tests for BoundingBoxDetection."""

from core.detections import BoundingBoxDetection
from core.geometry import Rectangle
import math
import pytest


@pytest.fixture
def rect():
    """A rectangle for testing."""
    return Rectangle.from_left_top_right_bottom(
        left=0.0, top=1.0, right=2.0, bottom=3.0
    )


def test_confidence(rect: Rectangle):
    """Tests that the correct confidence is returned."""
    bbox_det: BoundingBoxDetection = BoundingBoxDetection(
        rectangle=rect, category_scores=[0.3, 0.6, 0.1]
    )

    assert math.isclose(bbox_det.confidence, 0.6, rel_tol=1e-9)


def test_top_category_id_no_ties(rect: Rectangle):
    """Tests that the top_category_id property is correctly returned."""
    bbox_det: BoundingBoxDetection = BoundingBoxDetection(
        rectangle=rect, category_scores=[0.3, 0.6, 0.1]
    )

    assert bbox_det.top_category_id == 1


def test_top_category_id_with_ties(rect: Rectangle):
    """Tests that the top_category_id property is correctly returned in the presence of ties."""
    bbox_det: BoundingBoxDetection = BoundingBoxDetection(
        rectangle=rect, category_scores=[0.4, 0.2, 0.4]
    )

    assert bbox_det.top_category_id == 0
