"""Provides functions for extracting and determining the meaning of checkboxes."""

from utilities.annotations import BoundingBox
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

DATA_FILEPATH: Path = Path("..") / ".." / "data"
FILEPATH_TO_INTRAOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_centroids.json"
)
FILEPATH_TO_PREOP_POSTOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "preop_postop_centroids.json"
)
INTRAOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_INTRAOP_CENTROIDS, "r")
)
PREOP_POSTOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_PREOP_POSTOP_CENTROIDS, "r")
)


def find_checkbox_names(
    checkboxes: List[BoundingBox],
    centroids: Dict[str, Tuple[float, float]],
    threshold: float = 0.025,
) -> Dict[str, str]:
    """Finds the names of checkboxes.

    The checkboxes must be the locations after image registration, or else they will
    be incorrect due to the naive nature of this solution, which is to merely find
    the closest checkbox based on the centroid of a sample of 10 ground-truth
    checkboxes.

    Args:
        checkboxes (List[BoundingBox]):
            The checkboxes to classify. The 'category' attribute should be 'checked'
            or 'unchecked'.
        centroids (Dict[str, Tuple[float, float]]):
            The centroids of a sample of 10 checkboxes.
        threshold (float):
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
        distance_to_all_centroids: Dict[str, float] = {
            name: distance(ckbx.center, centroid)
            for (name, centroid) in centroids.items()
        }

        checkbox_too_far_from_any_centroid: bool = all(
            [dist > threshold for dist in list(distance_to_all_centroids.values())]
        )
        if checkbox_too_far_from_any_centroid:
            continue

        closest_checkbox_centroid: str = min(
            distance_to_all_centroids, key=distance_to_all_centroids.get
        )
        checkbox_values[closest_checkbox_centroid] = ckbx.category

    return checkbox_values


def find_interaoperative_checkbox_names(
    intraoperative_checkboxes: List[BoundingBox], threshold: float = 0.025
) -> Dict[str, str]:
    """Finds the names of intraoperative checkboxes."""
    return find_checkbox_names(intraoperative_checkboxes, INTRAOP_CENTROIDS, threshold)


def find_preoperative_checkbox_names(
    preoperative_checkboxes: List[BoundingBox], threshold: float = 0.025
) -> Dict[str, str]:
    """Finds the names of preoperative checkboxes."""
    return find_checkbox_names(
        preoperative_checkboxes, PREOP_POSTOP_CENTROIDS, threshold
    )
