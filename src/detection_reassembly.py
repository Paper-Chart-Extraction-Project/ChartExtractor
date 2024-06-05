""" """

from typing import List
from detections import Detection


def non_maximum_suppression(detections: List[Detection]):
    """Filters overlapping detections to only those which have the highest confidence."""
    pass


def intersection_over_union(detection_1: Detection, detection_2: Detection):
    """Divides two detection's area of intersection by their union area."""
    pass


def intersection_over_minimum(detection_1: Detection, detection_2: Detection):
    """Divides two detection's area of intersection over the area of the smaller of the two detections."""
    pass
