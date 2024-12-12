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


def get_category_or_space(bb):
    return bb.category if bb is not None else " "
