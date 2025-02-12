"""Consolidates all the functionality for extracting data from charts into one function."""

# Built-in imports
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

# Internal Imports
from extraction.blood_pressure_and_heart_rate import (
    extract_heart_rate_and_blood_pressure,
)
from extraction.checkboxes import extract_checkboxes
from extraction.extraction_utilities import detect_numbers
from extraction.inhaled_volatile import extract_inhaled_volatile
from extraction.intraoperative_digit_boxes import (
    extract_drug_codes,
    extract_ett_size,
    extract_surgical_timing,
)
from extraction.physiological_indicators import extract_physiological_indicators
from extraction.preoperative_postoperative_digit_boxes import (
    extract_preop_postop_digit_data,
)
from image_registration.homography import homography_transform
from label_clustering.cluster import Cluster
from label_clustering.clustering_methods import (
    cluster_kmeans,
    cluster_boxes,
    find_legend_locations,
)
from label_clustering.isolate_labels import isolate_blood_pressure_legend_bounding_boxes
from object_detection_models.ultralytics_yolov8 import UltralyticsYOLOv8
from object_detection_models.ultralytics_yolov11_pose import UltralyticsYOLOv11Pose
from object_detection_models.object_detection_model import ObjectDetectionModel
from utilities.annotations import BoundingBox
from utilities.detections import Detection
from utilities.detection_reassembly import (
    untile_detections,
    intersection_over_minimum,
    non_maximum_suppression,
)
from utilities.tiling import tile_image


CORNER_LANDMARK_NAMES: List[str] = [
    "anesthesia_start",
    "safety_checklist",
    "lateral",
    "units",
]


def label_studio_to_bboxes(
    path_to_json_data: Path,
    desired_im_width: int = 3300,
    desired_im_height: int = 2550,
) -> List[BoundingBox]:
    """
    Convert the json data from label studio to a list of BoundingBox objects
    Args:
        path_to_json_data (Path):
            Path to the json data from label studio
    Returns:
        List[BoundingBox]:
            List of BoundingBox objects
    """
    json_data: List[Dict] = json.loads(open(str(path_to_json_data)).read())
    return {
        sheet_data["data"]["image"].split("-")[-1]: [
            BoundingBox(
                category=label["value"]["rectanglelabels"][0],
                left=label["value"]["x"] / 100 * desired_im_width,
                top=label["value"]["y"] / 100 * desired_im_height,
                right=(label["value"]["x"] / 100 + label["value"]["width"] / 100)
                * desired_im_width,
                bottom=(label["value"]["y"] / 100 + label["value"]["height"] / 100)
                * desired_im_height,
            )
            for label in sheet_data["annotations"][0]["result"]
        ]
        for sheet_data in json_data
    }


def combine_dictionaries(dictionaries: List[Dict]):
    """Combines a list of dictionaries into one.

    Args:
        `dictionaries` (List[Dict]):
            A list of dictionaries to combine.

    Returns:
        A single dictionary with the contents of all the dictionaries.
    """
    combined_dictionary: Dict = dict()
    for dictionary in dictionaries:
        combined_dictionary.update(dictionary)
    return combined_dictionary
