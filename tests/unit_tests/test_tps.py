"""Tests tps module."""

# NOTE: this relies on the test data in the data folder, which is not included in the repository.
# If you would like to run these tests, please reach out to the author for the data.
# The main purpose is for demonstrating how to use the clustering methods.

# Standard Libs
import os
import sys
import json
from typing import Dict, List

# External Libraries
import cv2
import pytest
from PIL import Image

# Internal Libraries
# Add the directory containing the utilities module to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from utilities.annotations import BoundingBox
from image_registration.thin_plate_splining import transform_thin_plate_splines


@pytest.fixture
def test_data() -> Dict:
    """Test data for TestClustering."""
    with open(os.path.join("test_data", "yolo_data.json")) as json_file:
        data: Dict = list(json.load(json_file).values())[0]

    # Make category to id mapping
    category_to_id = {i: bb.split()[0] for i, bb in enumerate(data)}
    # Go through bounding boxes and change class to an integer while creating a mapping
    sheet_data = [
        f"{list(category_to_id.keys())[list(category_to_id.values()).index(bb.split()[0])]} {' '.join(bb.split()[1:])}"
        for bb in data
    ]

    # convert the YOLO data to Bounding Boxes
    data: List[BoundingBox] = [
        BoundingBox.from_yolo(yolo_bb, 800, 600, category_to_id)
        for yolo_bb in sheet_data
    ]
    return data


@pytest.fixture
def image() -> cv2:
    """Test data for TestClustering."""
    image = cv2.imread(
        os.path.join("test_data", "registered_images", "RC_0001_intraoperative.JPG")
    )
    resized_img = cv2.resize(image, (800, 600))
    return resized_img


class TestTPS:
    """Tests the tps method."""

    def test_tps(self, image, test_data):
        # get path to current image
        transformed_img = transform_thin_plate_splines(image, test_data)
        transformed_img = Image.fromarray(transformed_img)
        transformed_img.show()

        # Not sure how to test this ...

        # Check if the transformed image is not None
        assert transformed_img is not None
