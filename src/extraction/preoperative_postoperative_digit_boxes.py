"""Provides functions for extracting preoperative/postoperative handwritten digit data."""

# Built-in imports
from itertools import product
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Internal imports
from extraction.handwritten_digit_utils import compute_digit_distances_to_centroids
from utilities.annotations import BoundingBox
from utilities.detections import Detection


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "preop_postop_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)


def get_relevant_boxes(
    number_detections: List[Detection],
    keyword: str,
    im_width: int,
    im_height: int,
) -> Dict[str, BoundingBox]:
    """Gets the relevant BoundingBoxes from the list of all detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `keyword` (str):
            A word that appears in all the bounding boxes 'category' attributes.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the names of the relevant digit boxes to the closest bounding boxes.
    """
    number_detections: List[BoundingBox] = [det.annotation for det in number_detections]
    filtered_centroids: Dict[str, Tuple[float, float]] = {
        key: val for (key, val) in NUMBER_BOX_CENTROIDS.items() if keyword in key
    }
    values: Dict[str, BoundingBox] = compute_digit_distances_to_centroids(
        number_detections, filtered_centroids, im_width, im_height
    )
    return values


def get_category_or_space(bb: BoundingBox):
    """Gets the category of the bounding box, or return a space character."""
    return bb.category if bb is not None else " "


def extract_time_of_assessment(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    """Extracts the time of assessment data from the number detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the time of assessment to the hours and minutes they occured.
    """
    time_of_assessment_values = get_relevant_boxes(
        number_detections, "time_of_assessment", im_width, im_height
    )
    time_of_assessment: Dict[str, str] = dict()
    prefixes: List[str] = [
        f"time_of_assessment_{x}" for x in ["year", "month", "day", "hour", "min"]
    ]
    for prefix in prefixes:
        tens_place_val: Optional[int] = time_of_assessment_values.get(prefix + "_tens")
        ones_place_val: Optional[int] = time_of_assessment_values.get(prefix + "_ones")
        if None not in [tens_place_val, ones_place_val]:
            time_of_assessment[prefix] = str(tens_place_val.category) + str(
                ones_place_val.category
            )
    return time_of_assessment


def extract_age(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    """Extracts the age data from the number detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary with the patient's age.
    """
    age_values: Dict[str, int] = get_relevant_boxes(
        number_detections, "age", im_width, im_height
    )

    if any(
        [f"age_{place}" in age_values.keys() for place in ["hundreds", "tens", "ones"]]
    ):
        return {
            "age": "".join(
                [
                    get_category_or_space(age_values.get(f"age_{place}"))
                    for place in ["hundreds", "tens", "ones"]
                ]
            ).strip()
        }
    else:
        return dict()


def extract_height(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    """Extracts the height data from the number detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary with the patient's height.
    """
    height_values: Dict[str, int] = get_relevant_boxes(
        number_detections, "height", im_width, im_height
    )

    if any(
        [
            f"height_{place}" in height_values.keys()
            for place in ["hundreds", "tens", "ones"]
        ]
    ):
        return {
            "height": "".join(
                [
                    get_category_or_space(height_values.get(f"height_{place}"))
                    for place in ["hundreds", "tens", "ones"]
                ]
            ).strip()
        }
    else:
        return dict()


def extract_weight(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    """Extracts the weight data from the number detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary with the patient's weight.
    """
    weight_values: Dict[str, int] = get_relevant_boxes(
        number_detections, "weight", im_width, im_height
    )

    if any(
        [
            f"weight_{place}" in weight_values.keys()
            for place in ["hundreds", "tens", "ones"]
        ]
    ):
        return {
            "weight": "".join(
                [
                    get_category_or_space(weight_values.get(f"weight_{place}"))
                    for place in ["hundreds", "tens", "ones"]
                ]
            ).strip()
        }
    else:
        return dict()
