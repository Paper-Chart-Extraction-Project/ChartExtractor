"""Provides functions for extracting drug codes."""

# Built-in imports
import json
from pathlib import Path
from typing import Dict, List, Tuple

# External imports
import numpy as np

# Internal imports
from utilities.annotations import BoundingBox
from utilities.detections import Detection


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_single_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)
MAX_BOX_WIDTH, MAX_BOX_HEIGHT = (0.0174507, 0.0236938)


def extract_drug_codes(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, int]:
    """Extracts the drug code data from the number detections.

    Args:
        number_detections (List[Detection]):
            A list of Detection objects of handwritten digits.
        im_width (int):
            The width of the image the detections were made on.
        im_height (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping each line of the code section of the
        intraoperative record to the digits written on that line.
    """
    number_detections: List[BoundingBox] = [det.annotation for det in number_detections]
    drug_centroids: Dict[str, Tuple[float, float]] = {
        key: val for (key, val) in NUMBER_BOX_CENTROIDS.items() if "code_row" in key
    }
    drug_box_values: Dict[str, int] = compute_digit_distances_to_centroids(
        number_detections, drug_centroids, im_width, im_height
    )

    drug_codes: Dict[str, str] = dict()
    for ix in range(0, 11):
        if all(
            [
                f"code_row{str(ix).zfill(2)}_col{jx}" in drug_box_values.keys()
                for jx in range(0, 3)
            ]
        ):
            drug_codes[f"drug_row_{str(ix).zfill(2)}"] = "".join(
                [
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{0}"].category,
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{1}"].category,
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{2}"].category,
                ]
            )

    return drug_codes


def compute_digit_distances_to_centroids(
    number_detections: List[BoundingBox],
    centroids: Dict[str, Tuple[float, float]],
    im_width: int,
    im_height: int,
) -> Dict[str, int]:
    """Computes the distances between the digit detections and the number box centroids.

    Args:
        number_detections (List[BoundingBox]):
            Handwritten digit bounding boxes.
        centroids (Dict[str, Tuple[float, float]]):
            Tuples of floats that encode the centroid of a sample of single digit number boxes.
        im_width (int):
            The width of the image the detections were made on.
        im_height (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the names of the centroids to the closest bounding box.
    """
    euclidean_distance = lambda x1, y1, x2, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    normalize_box_loc = lambda center: (center[0] / im_width, center[1] / im_height)

    closest_boxes: Dict[str, int] = dict()
    for centroid_name, centroid in centroids.items():
        distance_dict: Dict[int, float] = {
            ix: euclidean_distance(
                *centroid, *normalize_box_loc(box.center, im_width, im_height)
            )
            for (ix, box) in enumerate(number_detections)
        }
        minimum_distance: float = min(distance_dict.values())
        if minimum_distance < MAX_BOX_WIDTH / 2:
            closest_boxes[centroid_name] = number_detections[
                min(distance_dict, key=distance_dict.get)
            ]
    return closest_boxes
