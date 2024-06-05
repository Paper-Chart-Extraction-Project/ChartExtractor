"""Tests for the detection_reassembly module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from typing import List
from annotations import BoundingBox
from detections import Detection
import detection_reassembly


@pytest.fixture()
def test_detections() -> List[Detection]:
    """Creates a short list of fake detections for testing purposes."""
    return [
        Detection(BoundingBox("Test", 0, 0, 1, 1), 1.0),
        Detection(BoundingBox("Test", 0.5, 0, 1, 1), 0.5),
        Detection(BoundingBox("Test", 2, 2, 3, 3), 0.75),
    ]


class TestIntersectionOverUnion:
    """Tests the intersection_over_union function."""

    def test_intersection_over_union(self, test_detections):
        """Tests the intersection_over_union function with typical inputs."""
        pass

    def test_intersection_over_union_no_overlap(self, test_detections):
        """Tests the intersection_over_union function with non-overlapping detections."""
        pass

    def test_intersection_over_union_total_overlap(self, test_detections):
        """Tests the intersection_over_union function with totally overlapping detections."""
        pass
