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
