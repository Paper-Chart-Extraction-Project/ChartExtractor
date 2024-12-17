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

    Finds the closest two legend values, then uses the distance between the detection and both
    of the closest values to find an approximate value in between.

    Args:
        `value_legend` (List[Cluster]):
            The named clusters which form the mmhg/bpm legend that runs vertically on the left
            side of the blood pressure and heart rate section.
        `keypoint_y` (float):
            The y value of the keypoint.

    Returns:
        The approximate value that the keypoint encodes in mmhg/bpm.
    """
    value_legend_centers: Dict[str, float] = {
        clust.label: clust.bounding_box.center[1] for clust in value_legend
    }
    distances: Dict[str, float] = {
        name: abs(legend_loc - keypoint_y)
        for (name, legend_loc) in value_legend_centers.items()
    }
    first_closest: str = min(distances, key=distances.get)
    first_distance: float = distances[first_closest]
    distances.pop(first_closest)
    second_closest: str = min(distances, key=distances.get)
    second_distance: float = distances[second_closest]
    total_dist: float = abs(
        value_legend_centers[first_closest] - value_legend_centers[second_closest]
    )
    first_weight: float = abs(first_distance - total_dist) / total_dist
    second_weight: float = abs(second_distance - total_dist) / total_dist
    first_val: float = first_weight * int(first_closest.split("_")[0])
    second_val: float = second_weight * int(second_closest.split("_")[0])
    weighted_value: float = first_val + second_val
    return int(weighted_value)


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
    data = dict()
    for det in detections:
        point: Tuple[float, float] = det.annotation.keypoint
        category: str = det.annotation.category
        suffix: str = "bpm" if category == "heart_rate" else "mmhg"
        timestamp: str = find_timestamp(time_clusters, point.x)
        value: int = find_value(value_clusters, point.y)
        if data.get(timestamp) is None:
            data[timestamp] = {category: f"{value}_{suffix}"}
        else:
            data[timestamp].update({category: f"{value}_{suffix}"})
    return data
