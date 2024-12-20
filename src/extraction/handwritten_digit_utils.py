"""Utilities for detecting and associating meaning to handwritten digits."""

# Built-in imports
from PIL import Image
from typing import Dict, List, Tuple

# External imports
import numpy as np

# Internal imports
from object_detection_models.object_detection_model import ObjectDetectionModel
from utilities.annotations import BoundingBox
from utilities.detections import Detection
from utilities.detection_reassembly import (
    intersection_over_minimum,
    non_maximum_suppression,
    untile_detections,
)
from utilities.tiling import tile_image


MAX_BOX_WIDTH, MAX_BOX_HEIGHT = (0.0174507, 0.0236938)


def compute_digit_distances_to_centroids(
    number_detections: List[BoundingBox],
    centroids: Dict[str, Tuple[float, float]],
    im_width: int,
    im_height: int,
) -> Dict[str, int]:
    """Computes the distances between the digit detections and the number box centroids.

    Args:
        `number_detections` (List[BoundingBox]):
            Handwritten digit bounding boxes.
        `centroids` (Dict[str, Tuple[float, float]]):
            Tuples of floats that encode the centroid of a sample of single digit number boxes.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the names of the centroids to the closest bounding box.
    """
    euclidean_distance = lambda x1, y1, x2, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    normalize_box_loc = lambda center: (center[0] / im_width, center[1] / im_height)

    closest_boxes: Dict[str, int] = dict()
    for centroid_name, centroid in centroids.items():
        distance_dict: Dict[int, float] = {
            ix: euclidean_distance(*centroid, *normalize_box_loc(box.center))
            for (ix, box) in enumerate(number_detections)
        }
        minimum_distance: float = min(distance_dict.values())
        if minimum_distance < MAX_BOX_WIDTH / 2:
            closest_boxes[centroid_name] = number_detections[
                min(distance_dict, key=distance_dict.get)
            ]
    return closest_boxes


def detect_numbers(
    image: Image.Image,
    detection_model: ObjectDetectionModel,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
    conf: float = 0.5,
) -> List[Detection]:
    """Detects handwritten digits on an image.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `detection_model` (ObjectDetectionModel):
            The digit detection model.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Returns:
        A list of handwritten digit detections on the image.
    """
    image_tiles: List[List[Image.Image]] = tile_image(
        image,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    detections: List[List[List[Detection]]] = [
        [detection_model(tile, verbose=False, conf=conf) for tile in row]
        for row in image_tiles
    ]
    detections: List[Detection] = untile_detections(
        detections,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    detections: List[Detection] = non_maximum_suppression(
        detections=detections,
        threshold=0.5,
        overlap_comparator=intersection_over_minimum,
    )
    return detections
