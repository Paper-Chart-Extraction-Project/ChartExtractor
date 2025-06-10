"""Tests the detections module's Detection class."""

# External Imports
import pytest

# Internal Imports
from ChartExtractor.utilities.annotations import BoundingBox, Keypoint, Point
from ChartExtractor.utilities.detections import Detection


class TestDetection:
    """Tests the Detection class."""

    def test_from_dict_bounding_box(self):
        """Tests the from_dict constructor with a bounding box."""
        det_dict = {
            "annotation": {
                "left": 0,
                "right": 1,
                "top": 2,
                "bottom": 3,
                "category": "Test",
            },
            "confidence": 0.8,
        }
        true_det = Detection(
            annotation=BoundingBox("Test", 0, 2, 1, 3),
            confidence=0.8,
        )
        assert Detection.from_dict(det_dict, BoundingBox) == true_det

    def test_from_dict_keypoint(self):
        """Tests the from_dict constructor with a keypoint."""
        det_dict = {
            "annotation": {
                "bounding_box": {
                    "left": 0,
                    "right": 1,
                    "top": 2,
                    "bottom": 3,
                    "category": "Test",
                },
                "keypoint": {
                    "x": 0.5,
                    "y": 2.25,
                },
            },
            "confidence": 0.8,
        }
        true_det = Detection(
            annotation=Keypoint(Point(0.5, 2.25), BoundingBox("Test", 0, 2, 1, 3)),
            confidence=0.8,
        )
        assert Detection.from_dict(det_dict, Keypoint) == true_det
    
    def test_from_dict_bounding_box(self):
        """Tests the to_dict method with a bounding box."""
        det = Detection(
            annotation=BoundingBox("Test", 0, 2, 1, 3),
            confidence=0.8,
        )
        true_dict = {
            "annotation": {
                "left": 0,
                "right": 1,
                "top": 2,
                "bottom": 3,
                "category": "Test",
            },
            "confidence": 0.8,
        }
        assert det.to_dict() == true_dict

    def test_to_dict_keypoint(self):
        """Tests the to_dict method with a keypoint."""
        det = Detection(
            annotation=Keypoint(Point(0.5, 2.25), BoundingBox("Test", 0, 2, 1, 3)),
            confidence=0.8,
        )
        true_dict = {
            "annotation": {
                "bounding_box": {
                    "left": 0,
                    "right": 1,
                    "top": 2,
                    "bottom": 3,
                    "category": "Test",
                },
                "keypoint": {
                    "x": 0.5,
                    "y": 2.25,
                },
            },
            "confidence": 0.8,
        }
        
        assert det.to_dict() == true_dict

