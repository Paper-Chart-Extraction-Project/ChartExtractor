"""This file provides functions for extracting blood pressure and heart rate data."""

# Built-in imports
from typing import Dict, List, Tuple

# Internal imports
from label_clustering.cluster import Cluster
from utilities.annotations import BoundingBox
from utilities.detections import Detection


def find_timestamp(time_legend: List[Cluster], keypoint_x: float) -> str:
    """Given a keypoint on a blood pressure or heart rate detection, finds the timestamp.

    Args:
        `time_legend` (List[Cluster]):
            The named clusters which form the timestamp legend that runs horizontally on the top
            side of the blood pressure and heart rate section.
        `keypoint_x` (float):
            The x value of the keypoint.

    Returns:
        The label of the closest timestamp cluster.
    """
    time_legend_centers: Dict[str, float] = {
        clust.label: clust.bounding_box.center[0] for clust in time_legend
    }
    distances: Dict[str, float] = {
        name: abs(legend_loc - keypoint_x)
        for (name, legend_loc) in time_legend_centers.items()
    }
    return min(distances, key=distances.get)


def find_value(value_legend: List[Cluster], keypoint_y: float) -> int:
    """Given a keypoint on a blood pressure or heart rate detection, finds the in mmhg/bpm value.

    Args:
        `value_legend` (List[Cluster]):
            The named clusters which form the mmhg/bpm legend that runs vertically on the left
            side of the blood pressure and heart rate section.
        `keypoint_y` (float):
            The y value of the keypoint.

    Returns:
        The approximate value that the keypoint encodes in mmhg/bpm.
    """
    pass


def extract_heart_rate_and_blood_pressure(
    detections: List[Detection],
    time_clusters: List[Cluster],
    value_clusters: List[Cluster],
) -> Dict[str, Dict[str, str]]:
    """Extracts the heart rate and blood pressure data from the detections.

    Args:
        `detections` (List[Detection]):
            The keypoint detections of the systolic, diastolic, and heart rate markings.
        `time_clusters` (List[Cluster]):
            The clusters corresponding to the timestamps.
        `value_clusters` (List[Cluster]):
            The clusters corresponding to the mmhg and bpm values.

    Returns:
        A dictionary mapping each timestamp to the systolic, diastolic, and heart rate reading
        that was recorded at that time.
    """
    pass
