"""This module defines functions for reassembling tiled detections."""

# Built-in Imports
from typing import Callable, List, Tuple

# Internal Imports
from ..utilities.detections import Detection
from ..utilities.tiling import correct_annotation_coords


def compute_area(box: Tuple[float, float, float, float]):
    """Computes the area of a rectangle.

    Args:
        `box` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of a rectangle.

    Returns:
        The area of the rectangle.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_intersection_area(
    box_1: Tuple[float, float, float, float], box_2: Tuple[float, float, float, float]
) -> float:
    """Computes the area of the intersection of two rectangle.

    Args:
        `box_1` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of the first rectangle.
        `box_2` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of the second rectangle.

    Returns:
        The area of the intersection of the two rectangles box_1 and box_2.
    """
    intersection_left = max(box_1[0], box_2[0])
    intersection_top = max(box_1[1], box_2[1])
    intersection_right = min(box_1[2], box_2[2])
    intersection_bottom = min(box_1[3], box_2[3])
    if intersection_right < intersection_left or intersection_bottom < intersection_top:
        return 0
    intersection_area = compute_area(
        [intersection_left, intersection_top, intersection_right, intersection_bottom]
    )
    return intersection_area


def intersection_over_minimum(detection_1: Detection, detection_2: Detection) -> float:
    """Calculates the Intersection over Minimum (IoM) between two detections.

    This function calculates the area of overlap between the two bounding boxes of two
    detection objects and divides it by the area of the smaller detection.

    Args:
        `detection_1` (Detection):
            A Detection object representing the first detection.
        `detection_2` (Detection):
            A Detection object representing the second detection.

    Returns:
        A float value between 0.0 and 1.0 representing the IoM between the two detections.
    """
    box_1, box_2 = detection_1.annotation.box, detection_2.annotation.box
    intersection_area = compute_intersection_area(box_1, box_2)
    minimum_area = min(compute_area(box_1), compute_area(box_2))
    return intersection_area / minimum_area


def intersection_over_union(detection_1: Detection, detection_2: Detection) -> float:
    """Calculates the Intersection over Union (IoU) between two detections.

    This function calculates the area of overlap between the bounding boxes of two
    detection objects and divides it by the total area covered by their bounding boxes.

    Args:
        `detection_1` (Detection):
            A Detection object representing the first detection.
        `detection_2` (Detection):
            A Detection object representing the second detection.

    Returns:
        A float value between 0.0 and 1.0 representing the IoU between the two detections.
    """
    box_1, box_2 = detection_1.annotation.box, detection_2.annotation.box
    intersection_area = compute_intersection_area(box_1, box_2)
    union_area = compute_area(box_1) + compute_area(box_2) - intersection_area
    return intersection_area / union_area


def non_maximum_suppression(
    detections: List[Detection],
    threshold: float = 0.5,
    overlap_comparator: Callable[
        [Detection, Detection], float
    ] = intersection_over_union,
) -> List[Detection]:
    """Applies Non-Maximum Suppression (NMS) to a list of detections.

    This function filters a list of detections to remove overlapping bounding boxes
    based on their confidence scores. It keeps only the detections with the highest
    confidence scores for each object.

    Args:
        `detections` (List[Detection]):
            A list of `Detection` objects representing the detections to be filtered.
        `threshold` (float):
            A float value between 0.0 and 1.0, representing the minimum IoU (Intersection
            over Union) threshold for discarding detections considered to overlap with
            a higher-confidence detection. (default: 0.5)
        `overlap_comparator`:
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


def untile_detections(
    tiled_detections: List[List[List[Detection]]],
    tile_width: int,
    tile_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
) -> List[Detection]:
    """Squashes multiple detection lists into one and corrects their location on the main image.

    Args:
        `tiled_detections` (List[List[Detection]]):
            A list of detections made on tiles.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.
    """
    generate_tile_left = lambda ix: int(ix * tile_width * horizontal_overlap_ratio)
    generate_tile_top = lambda iy: int(iy * tile_height * vertical_overlap_ratio)
    flatten_list = lambda l: [item for sublist in l for item in sublist]

    untiled_detections: List[List[List[Detection]]] = [
        [
            [
                Detection(
                    correct_annotation_coords(
                        detection.annotation,
                        generate_tile_left(ix),
                        generate_tile_top(iy),
                        "tile_to_image",
                    ),
                    detection.confidence,
                )
                for detection in tile_detections
            ]
            for ix, tile_detections in enumerate(detection_row)
        ]
        for iy, detection_row in enumerate(tiled_detections)
    ]
    untiled_detections: List[Detection] = flatten_list(flatten_list(untiled_detections))
    return untiled_detections
