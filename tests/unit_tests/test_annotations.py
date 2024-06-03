""" """

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
import annotations


class TestBoundingBox:
    """Tests the BoundingBox class."""

    def test_init(self):
        """Tests the init function with valid parameters."""
        pass
