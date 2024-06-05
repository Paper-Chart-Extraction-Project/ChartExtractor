""" """

from typing import Callable, List
from detections import Detection


def intersection_over_union(detection_1: Detection, detection_2: Detection):
    """Divides two detection's area of intersection by their union area."""
    pass


def intersection_over_minimum(detection_1: Detection, detection_2: Detection):
    """Divides two detection's area of intersection over the area of the smaller of the two detections."""
    pass


def non_maximum_suppression(
    detections: List[Detection],
    threshold: float = 0.5,
    overlap_comparator: Callable[
        [Detection, Detection], float
    ] = intersection_over_union,
):
    """Applies Non-Maximum Suppression (NMS) to a list of detections.

    This function filters a list of detections to remove overlapping bounding boxes
    based on their confidence scores. It keeps only the detections with the highest
    confidence scores for each object.

    Args:
        detections:
            A list of `Detection` objects representing the detections to be filtered.
        threshold:
            A float value between 0.0 and 1.0, representing the minimum IoU (Intersection
            over Union) threshold for discarding detections considered to overlap with
            a higher-confidence detection. (default: 0.5)
        overlap_comparator:
            A callable function that takes two `Detection` objects as arguments and returns
            a float value representing the IoU (overlap) between their bounding boxes.
            (default: `intersection_over_union` function)

    Returns:
        A list of `Detection` objects containing the filtered detections after applying NMS.
    """
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    ix = 0
    while ix < len(detections):
        jx = ix + 1
        while jx < len(detections):
            if overlap_comparator(detections[ix], detections[jx]) >= threshold:
                del [detections[jx]]
            else:
                jx += 1
        ix += 1
    return detections
