"""This file provides functions for extracting blood pressure and heart rate data."""

# Built-in imports
from typing import Dict, List, Tuple

# Internal imports
from ..label_clustering.cluster import Cluster
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection


def find_timestamp(legend: Dict[str, Tuple[float, float]], keypoint_x: float) -> str:
    """Given a keypoint on a blood pressure or heart rate detection, finds the timestamp.

    Args:
        `legend` (Dict[str, Tuple[float, float]]):
            The dictionary that maps the name of legend entries to their locations on the image.
        `keypoint_x` (float):
            The x value of the keypoint.

    Returns:
        The label of the closest timestamp cluster.
    """
    time_legend: Dict[str, Tuple[float, float]] = {
        k:v for (k, v) in legend.items() if "_mins" in k
    }
    distances: Dict[str, float] = {
        name: abs(legend_loc[0] - keypoint_x)
        for (name, legend_loc) in time_legend.items()
    }
    return min(distances, key=distances.get)


def find_value(legend: Dict[str, Tuple[float, float]], keypoint_y: float) -> int:
    """Given a keypoint on a blood pressure or heart rate detection, finds the in mmhg/bpm value.

    Finds the closest two legend values, then uses the distance between the detection and both
    of the closest values to find an approximate value in between.

    Args:
        `legend` (Dict[str, Tuple[float, float]]):
            The dictionary that maps the name of legend entries to their locations on the image.
        `keypoint_y` (float):
            The y value of the keypoint.

    Returns:
        The approximate value that the keypoint encodes in mmhg/bpm.
    """
    value_legend: Dict[str, float] = {
        k:v for (k, v) in legend.items() if "_mmhg" in k
    }
    distances: Dict[str, float] = {
        name: abs(legend_loc[1] - keypoint_y)
        for (name, legend_loc) in value_legend.items()
    }
    first_closest: str = min(distances, key=distances.get)
    distances.pop(first_closest)
    second_closest: str = min(distances, key=distances.get)
    total_dist: float = abs(
        value_legend[first_closest][1] - value_legend[second_closest][1]
    )
    smaller_of_two_values = min(
        [first_closest, second_closest], key=lambda leg: int(leg.split("_")[0])
    )
    fractional_component = (
        abs(value_legend[smaller_of_two_values][1] - keypoint_y) / total_dist
    ) * 10
    return int(smaller_of_two_values.split("_")[0]) + int(fractional_component)


def extract_heart_rate_and_blood_pressure(
    detections: List[Detection],
    legend: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, str]]:
    """Extracts the heart rate and blood pressure data from the detections.

    Args:
        `detections` (List[Detection]):
            The keypoint detections of the systolic, diastolic, and heart rate markings.
        `legend` (Dict[str, Tuple[float, float]]):
            The dictionary that maps the name of legend entries to their locations on the image.

    Returns:
        A dictionary mapping each timestamp to the systolic, diastolic, and heart rate reading
        that was recorded at that time.
    """
    
    def filter_detections_outside_bp_and_hr_area(detections):
        leftmost_point: float = min([point[0] for point in legend.values()])
        topmost_point: float = min([point[1] for point in legend.values()])
        rightmost_point: float = max([point[0] for point in legend.values()])
        bottommost_point: float = max([point[1] for point in legend.values()])

        return list(
            filter(
                lambda d: all(
                    [
                        d.annotation.bottom > topmost_point,
                        d.annotation.top < bottommost_point,
                        d.annotation.right > leftmost_point,
                        d.annotation.left < rightmost_point,
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
        timestamp: str = find_timestamp(legend, point.x)
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
            value: int = find_value(legend, point.y)
            data[timestamp][category] = f"{value}_{suffix}"
    return data
