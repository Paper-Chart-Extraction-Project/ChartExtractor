"""Tests for the detection_reassembly module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
import detection_reassembly


@pytest.fixture()
def test_detections():
    """Creates a short list of fake detections for testing purposes."""
    pass
