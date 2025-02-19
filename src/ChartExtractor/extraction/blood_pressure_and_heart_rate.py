"""This file provides functions for extracting blood pressure and heart rate data."""

# Built-in imports
from typing import Dict, List, Tuple

# Internal imports
from .label_clustering.cluster import Cluster
from .utilities.annotations import BoundingBox
from .utilities.detections import Detection


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
    distances.pop(first_closest)
    second_closest: str = min(distances, key=distances.get)
    total_dist: float = abs(
        value_legend_centers[first_closest] - value_legend_centers[second_closest]
    )
    smaller_of_two_values = min(
        [first_closest, second_closest], key=lambda leg: int(leg.split("_")[0])
    )
    fractional_component = (
        abs(value_legend_centers[smaller_of_two_values] - keypoint_y) / total_dist
    ) * 10
    return int(smaller_of_two_values.split("_")[0]) + int(fractional_component)


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

    def filter_detections_outside_bp_and_hr_area(detections):
        return list(
            filter(
                lambda d: all(
                    [
                        d.annotation.bottom
                        > min(vc.bounding_box.top for vc in value_clusters),
                        d.annotation.top
                        < max(vc.bounding_box.bottom for vc in value_clusters),
                        d.annotation.left
                        > min(tc.bounding_box.left for tc in time_clusters),
                        d.annotation.right
                        < max(tc.bounding_box.right for tc in time_clusters),
                    ]
                ),
                detections,
            )
        )

    data = dict()
    # filter out any detection outside of the bp and hr area
    detections = filter_detections_outside_bp_and_hr_area(detections)
    for det in detections:
        point: Tuple[float, float] = det.annotation.keypoint
        category: str = det.annotation.category
        timestamp: str = find_timestamp(time_clusters, point.x)
        if data.get(timestamp) is None:
            data[timestamp] = {category: det}
        elif data[timestamp].get(category) is None:
            data[timestamp].update({category: det})
        elif data[timestamp][category].confidence < det.confidence:
            data[timestamp][category] = det
        else:
            pass

    for timestamp in data.keys():
        for category in data[timestamp].keys():
            point: Tuple[float, float] = data[timestamp][category].annotation.keypoint
            suffix: str = "bpm" if category == "heart_rate" else "mmhg"
            value: int = find_value(value_clusters, point.y)
            data[timestamp][category] = f"{value}_{suffix}"
    return data
