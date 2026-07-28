"""A series of tests for BoundingBoxWithKeypoints."""

from core.annotations import (
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point, Rectangle
import pytest
from typing import List
import warnings


@pytest.fixture
def test_bounding_box():
    return BoundingBox(
        rectangle=Rectangle.from_left_top_right_bottom(
            left=0.0, top=1.0, right=2.0, bottom=3.0
        ),
        category="test",
    )


class TestWarnKeypointsNotContainedInBox:
    """A test class for the warn_keypoints_not_contained_in_box model validator."""

    def test_all_keypoints_in_box(self, test_bounding_box: BoundingBox):
        """Tests that no warning is emitted when all the keypoints are in the box."""
        test_keypoints: List[Keypoint] = [
            Keypoint(
                point=Point(x=0.5, y=1.2),
                category="0",
                visibility=VisibilityStatus.VISIBLE,
            ),
            Keypoint(
                point=Point(x=1.7, y=2.0),
                category="1",
                visibility=VisibilityStatus.VISIBLE,
            ),
            Keypoint(
                point=Point(x=1.9, y=1.5),
                category="2",
                visibility=VisibilityStatus.OCCLUDED,
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            BoundingBoxWithKeypoints(
                bounding_box=test_bounding_box, keypoints=test_keypoints
            )

    def test_keypoint_not_in_box(self, test_bounding_box: BoundingBox):
        """Tests that a warning is emitted when a keypoint is not in the box."""
        test_keypoints: List[Keypoint] = [
            Keypoint(
                point=Point(x=0.5, y=1.2),
                category="0",
                visibility=VisibilityStatus.VISIBLE,
            ),
            Keypoint(
                point=Point(x=1.7, y=2.0),
                category="1",
                visibility=VisibilityStatus.VISIBLE,
            ),
            Keypoint(
                point=Point(x=1.9, y=3.5),
                category="2",
                visibility=VisibilityStatus.OCCLUDED,
            ),
        ]
        with pytest.warns(UserWarning, match="not contained in"):
            BoundingBoxWithKeypoints(
                bounding_box=test_bounding_box, keypoints=test_keypoints
            )


class TestWarnNoKeypoints:
    """A test class for the warn_no_keypoints model validator."""

    def test_keypoints_present(self, test_bounding_box: BoundingBox):
        """Tests that a warning is not emitted if keypoints are present."""
        test_keypoints: List[Keypoint] = [
            Keypoint(
                point=Point(x=0.5, y=1.2),
                category="0",
                visibility=VisibilityStatus.VISIBLE,
            )
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            BoundingBoxWithKeypoints(
                bounding_box=test_bounding_box, keypoints=test_keypoints
            )

    def test_keypoints_not_present(self, test_bounding_box: BoundingBox):
        """Tests that a warning is emitted if keypoints are not present."""
        with pytest.warns(UserWarning, match="has no keypoints"):
            BoundingBoxWithKeypoints(bounding_box=test_bounding_box, keypoints=[])
