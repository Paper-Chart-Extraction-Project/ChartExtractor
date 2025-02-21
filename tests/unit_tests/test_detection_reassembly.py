"""Tests for the detection_reassembly module."""

# Built-in Imports
from typing import List

# External Imports
import pytest

# Internal Imports
from ChartExtractor.utilities.annotations import BoundingBox
from ChartExtractor.utilities.detections import Detection
from ChartExtractor.utilities.tiling import tile_annotations
from ChartExtractor.utilities import detection_reassembly


@pytest.fixture()
def test_detections() -> List[Detection]:
    """Creates a short list of fake detections for testing purposes."""
    return [
        Detection(BoundingBox("Test", 0, 0, 1, 1), 1.0),
        Detection(BoundingBox("Test", 0.5, 0, 1, 1), 0.5),
        Detection(BoundingBox("Test", 2, 2, 3, 3), 0.75),
        Detection(BoundingBox("Test", 2.25, 2.25, 2.75, 2.75), 0.9),
    ]


def test_compute_area(test_detections):
    """Tests the compute_area function."""
    area_1 = detection_reassembly.compute_area(test_detections[0].annotation.box)
    area_2 = detection_reassembly.compute_area(test_detections[1].annotation.box)
    area_3 = detection_reassembly.compute_area(test_detections[2].annotation.box)
    area_4 = detection_reassembly.compute_area(test_detections[3].annotation.box)

    assert area_1 == 1
    assert area_2 == 0.5
    assert area_3 == 1
    assert area_4 == 0.25


class TestComputeIntersectionArea:
    """Tests the compute_intersection_area function."""

    def test_typical_inputs(self, test_detections):
        """Tests the compute_intersection_area function with normal inputs."""
        intersection_area = detection_reassembly.compute_intersection_area(
            test_detections[0].annotation.box, test_detections[1].annotation.box
        )
        assert intersection_area == 0.5

    def test_inscribed(self, test_detections):
        """Tests the compute_intersection_area function where one rectangle is within the other."""
        intersection_area = detection_reassembly.compute_intersection_area(
            test_detections[2].annotation.box, test_detections[3].annotation.box
        )
        assert intersection_area == 0.25

    def test_no_overlap(self, test_detections):
        """Tests the compute_intersection_area function with no overlap."""
        intersection_area = detection_reassembly.compute_intersection_area(
            test_detections[0].annotation.box, test_detections[2].annotation.box
        )
        assert intersection_area == 0


class TestIntersectionOverMinimum:
    """Tests the intersection_over_minimum function."""

    def test_typical_inputs(self, test_detections):
        """Tests the intersection_over_minimum function with normal inputs."""
        created_iom = detection_reassembly.intersection_over_minimum(
            test_detections[0], test_detections[1]
        )
        true_iom = 1
        assert created_iom == true_iom

    def test_inputs_with_no_overlap(self, test_detections):
        """Tests the intersection_over_minimum function with no overlap."""
        created_iom = detection_reassembly.intersection_over_minimum(
            test_detections[0], test_detections[2]
        )
        true_iom = 0
        assert created_iom == true_iom

    def test_inputs_with_total_overlap(self, test_detections):
        """Tests the intersection_over_minimum function with totally overlapping detections."""
        created_iom = detection_reassembly.intersection_over_minimum(
            test_detections[0], test_detections[0]
        )
        true_iom = 1
        assert created_iom == true_iom


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


def test_untile_detections(test_detections):
    """Tests the untile_detections function."""
    tiled_annotations = tile_annotations(test_detections, 4, 4, 2, 2, 0.5, 0.5)
    tiled_detections = [
        [
            [Detection(ann, 0.5) for ann in tile_annotations]
            for tile_annotations in row_annotations
        ]
        for row_annotations in tiled_annotations
    ]
    untiled_detections = untiled_detections(tiled_detections, 2, 2, 0.5, 0.5)
    assert test_detections == untiled_detections
