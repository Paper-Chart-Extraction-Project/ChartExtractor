"""Provides functions for extracting and determining the meaning of checkboxes."""

# Built-in Imports
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

# Internal Imports
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection

# External Imports
import numpy as np


DATA_FILEPATH: Path = Path(__file__).parents[1] / "data"
FILEPATH_TO_INTRAOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_checkbox_centroids.json"
)
FILEPATH_TO_PREOP_POSTOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "preop_postop_checkbox_centroids.json"
)
INTRAOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_INTRAOP_CENTROIDS, "r")
)
PREOP_POSTOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_PREOP_POSTOP_CENTROIDS, "r")
)


def extract_checkboxes(
    detections: List[Detection],
    side: Literal["intraoperative", "preoperative"],
    image_width: int,
    image_height: int,
) -> Dict[str, str]:
    """Extracts checkbox data from an image of a chart.

    Args:
        detections (List[Detection]):
            The detected checkboxes.
        side (Literal["intraoperative", "preoperative"]):
            The side of the chart.
        image_width (int):
            The original image's width.
        image_height (int):
            The original image's height.
    Returns:
        A dictionary mapping the name of checkboxes to "checked" or "unchecked".
    """
    if side.lower() == "intraoperative":
        centroids = INTRAOP_CENTROIDS
    elif side.lower() == "preoperative":
        centroids = PREOP_POSTOP_CENTROIDS
    else:
        raise ValueError(
            f'Invalid selection for side. Must be one of ["intraoperative", "preoperative"], value supplied was {side}'
        )

    checkbox_bboxes: List[BoundingBox] = [det.annotation for det in detections]
    names: Dict[str, str] = find_checkbox_names(
        checkbox_bboxes, centroids, image_width, image_height
    )
    return names


def find_checkbox_names(
    checkboxes: List[BoundingBox],
    centroids: Dict[str, Tuple[float, float]],
    image_width: int,
    image_height: int,
    threshold: float = 0.025,
) -> Dict[str, str]:
    """Finds the names of checkboxes.

    The checkboxes must be the locations after image registration, or else they will
    be incorrect due to the naive nature of this solution, which is to merely find
    the closest checkbox based on the centroid of a sample of 10 ground-truth
    checkboxes.

    Args:
        `checkboxes` (List[BoundingBox]):
            The checkboxes to classify. The 'category' attribute should be 'checked'
            or 'unchecked'.
        `centroids` (Dict[str, Tuple[float, float]]):
            The centroids of a sample of 10 checkboxes.
        `threshold` (float):
            The threshold that determines how far a centroid can be before it
            is totally ruled out. If all checkbox centroids are more than the
            threshold away, there is no associated name for that checkbox.
            Defaults to 2.5% of the image's width and height.

    Returns:
        A dictionary that maps the name of the checkbox to 'checked' or 'unchecked'.
    """

    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Computes euclidean distance."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    checkbox_values: Dict[str, str] = dict()
    for ckbx in checkboxes:
        center = ckbx.center[0] / image_width, ckbx.center[1] / image_height
        distance_to_all_centroids: Dict[str, float] = {
            name: distance(center, centroid) for (name, centroid) in centroids.items()
        }
        checkbox_too_far_from_any_centroid: bool = all(
            [dist > threshold for dist in list(distance_to_all_centroids.values())]
        )
        closest_checkbox_centroid: str = min(
            distance_to_all_centroids, key=distance_to_all_centroids.get
        )
        checkbox_values[closest_checkbox_centroid] = ckbx.category

    return checkbox_values
