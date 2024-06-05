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
        Detection(BoundingBox("Test", 2.25, 2.25, 2.75, 2.75), 0.9),
    ]


class TestIntersectionOverUnion:
    """Tests the intersection_over_union function."""

    def test_typical_inputs(self, test_detections):
        """Tests the intersection_over_union function with typical inputs."""
        created_iou = detection_reassembly.intersection_over_union(
            test_detections[0], test_detections[1]
        )
        true_iou = 0.5
        assert created_iou == true_iou

    def test_inputs_with_no_overlap(self, test_detections):
        """Tests the intersection_over_union function with non-overlapping detections."""
        created_iou = detection_reassembly.intersection_over_union(
            test_detections[0], test_detections[2]
        )
        true_iou = 0
        assert created_iou == true_iou

    def test_inputs_with_total_overlap(self, test_detections):
        """Tests the intersection_over_union function with totally overlapping detections."""
        created_iou = detection_reassembly.intersection_over_union(
            test_detections[0], test_detections[0]
        )
        true_iou = 1
        assert created_iou == true_iou


class TestNonMaximumSuppression:
    """Tests the non_maximum_suppression function."""

    def test_typical_inputs(self, test_detections):
        """Tests the non_maximum_suppression with typical inputs."""
        created_filtered_detections = detection_reassembly.non_maximum_suppression(
            test_detections
        )
        true_filtered_detections = [
            test_detections[0],
            test_detections[3],
            test_detections[2],
        ]
        assert created_filtered_detections == true_filtered_detections

    def test_no_inputs(self):
        """Tests the non_maximum_suppression function with no detections."""
        created_filtered_detections = detection_reassembly.non_maximum_suppression([])
        assert created_filtered_detections == []
