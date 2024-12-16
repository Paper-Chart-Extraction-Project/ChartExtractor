"""This file provides functions for extracting blood pressure and heart rate data."""

# Built-in imports
from typing import Dict, List, Tuple

# Internal imports
from label_clustering.cluster import Cluster
from utilities.annotations import BoundingBox
from utilities.detections import Detection


def find_timestamp(time_legend: List[Cluster], keypoint_x: float) -> str:
    """ """
    pass


def find_value(value_legend: List[Cluster], keypoint_y: float) -> int:
    """ """
    pass


def extract_heart_rate_and_blood_pressure(
    detections: List[Detection],
    time_clusters: List[Cluster],
    value_clusters: List[Cluster],
) -> Dict[str, Dict[str, str]]:
    """ """
    pass
