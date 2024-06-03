"""Tests for the annotations module."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

import pytest
from annotations import BoundingBox, Keypoint, Point


class TestBoundingBox:
    """Tests the BoundingBox class."""

    # Init
    def test_init(self):
        """Tests the init function with valid parameters."""
        BoundingBox("Test", 0, 0, 1, 1)

    # from_yolo
    def test_from_yolo(self):
        """Tests the from_yolo constructor."""
        true_bbox = BoundingBox("Test", 0, 0, 1, 1)
        yolo_line = "0 0.25 0.25 0.5 0.5"
        image_width = 2
        image_height = 2
        id_to_category = {0: "Test"}
        created_bbox = BoundingBox.from_yolo(
            yolo_line, image_width, image_height, id_to_category
        )
        assert true_bbox == created_bbox

    def test_from_yolo_category_not_in_id_to_category_dict(self):
        """Tests the from_yolo constructor where the supplied id is not in the id_to_category dictionary."""
        yolo_line = "0 0.25 0.25 0.5 0.5"
        image_width = 2
        image_height = 2
        id_to_category = {1: "Test"}
        with pytest.raises(
            ValueError, match="not found in the id_to_category dictionary"
        ):
            BoundingBox.from_yolo(yolo_line, image_width, image_height, id_to_category)

    # from_coco
    def test_from_coco(self):
        """Tests the from_coco constructor."""
        true_bbox = BoundingBox("Test", 0, 0, 1, 1)
        coco_annotation = {
            "id": 0,
            "image_id": 0,
            "category_id": 0,
            "bbox": [0, 0, 1, 1],
        }
        categories = [{"id": 0, "name": "Test"}]
        created_bbox = BoundingBox.from_coco(coco_annotation, categories)
        assert true_bbox == created_bbox

    def test_from_coco_category_not_found_in_(self):
        """Tests the from_coco constructor where the supplied category is not in the list of category dictionaries."""
        coco_annotation = {
            "id": 0,
            "image_id": 0,
            "category_id": 0,
            "bbox": [0, 0, 1, 1],
        }
        categories = [{"id": 1, "name": "Test"}]
        with pytest.raises(ValueError, match="not found in the categories list"):
            BoundingBox.from_coco(coco_annotation, categories)

    # validate_box_values
    def test_validate_box_values_left_greater_than_right(self):
        """Tests the validate_box_values classmethod with invalid parameters (left > right)."""
        with pytest.raises(ValueError, match="left side greater than its right side"):
            BoundingBox("Test", 1, 0, 0, 1)

    def test_validate_box_values_top_greater_than_bottom(self):
        """Tests the validate_box_values classmethod with invalid parameters (top > bottom)."""
        with pytest.raises(ValueError, match="top side greater than its bottom side"):
            BoundingBox("Test", 0, 1, 1, 0)

    def test_validate_box_values_left_eq_right(self):
        """Tests the validate_box_values classmethod with degenerate rectangle parameters (left == right)."""
        with pytest.warns(UserWarning, match="left side equals its right side"):
            BoundingBox("Test", 0, 0, 0, 1)

    def test_validate_box_values_top_eq_bottom(self):
        """Tests the validate_box_values classmethod with degenerate rectangle parameters (top == bottom)."""
        with pytest.warns(UserWarning, match="top side equals its bottom side"):
            BoundingBox("Test", 0, 0, 1, 0)

    def test_validate_box_values_left_eq_right_top_eq_bottom(self):
        """Tests the validate_box_values classmethod with degernate rectangle parameters (left == right == top == bottom)."""
        with pytest.warns(UserWarning, match="box's parameters are equal"):
            BoundingBox("Test", 0, 0, 0, 0)

    # Center
    def test_center(self):
        """Tests the 'center' property."""
        bbox = BoundingBox("Test", 0, 0, 1, 1)
        assert (0.5, 0.5) == bbox.center

    # Box
    def test_box(self):
        """Tests the 'box' property."""
        bbox = BoundingBox("Test", 0, 0, 1, 1)
        assert [0, 0, 1, 1] == bbox.box

    # to_yolo
    def test_to_yolo(self):
        """Tests the to_yolo method."""
        pass
