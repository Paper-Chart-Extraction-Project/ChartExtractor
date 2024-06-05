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
    ]
