"""Provides functions for extracting the physiological indicators from the chart."""

# Built-in imports
from itertools import pairwise
from typing import Dict, List, Tuple

# External imports
import numpy as np

# Internal imports
from label_clustering.cluster import Cluster
from utilities.annotations import BoundingBox
from utilities.detections import Detection


PHYSIO_LANDMARK_NAMES: List[str] = [
    "spo2",
    "etco2",
    "fio2",
    "temperature",
    "tidal_volume",
    "respiratory_rate",
    "urine_output",
    "blood_loss",
]


def extract_physiological_indicators(
    digit_detections: List[Detection],
    legend_locations: Dict[str, Tuple[float, float]],
    document_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, Dict[str, List[int]]]:
    """Extracts all of the physiological indicator data from the number detections.

    Args:
        `number_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `legend_locations` (Dict[str, Tuple[float, float]]):
            The location of timestamps and mmhg/bpm values on the legend.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the name of the physiological indicator with a dictionary
        that maps timestamps to values.
    """
    pass


def find_indicator_for_bbox(
    bbox: BoundingBox, document_detections: List[Detection], im_width: int = 1
) -> str:
    """Determines which physiological indicator the bounding box belongs to.

    Args:
        `bbox` (BoundingBox):
            The bounding box in question.
        `document_detections` (List[Detection]):
            All of the document landmark detections.
        `im_width` (int):
            The width of the image.

    Returns:
        A string showing which physiological indicator the box belongs to.
    """
    physio_landmarks: List[Detection] = list(
        filter(
            lambda det: all(
                [
                    (det.annotation.category in PHYSIO_LANDMARK_NAMES),
                    (det.annotation.center[0] < 0.5 * im_width),
                ]
            ),
            document_detections,
        )
    )
    physio_landmarks: List[BoundingBox] = [det.annotation for det in physio_landmarks]
    distances: Dict[str, float] = {
        pl.category: abs(pl.center[1] - bbox.center[1]) for pl in physio_landmarks
    }
    return min(distances, key=distances.get)
