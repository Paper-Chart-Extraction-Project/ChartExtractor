"""Provides functions for extracting and determining the meaning of checkboxes."""

# Built-in Imports
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Literal, Tuple

# Internal Imports
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import (
    untile_detections,
    non_maximum_suppression,
    intersection_over_minimum,
)
from ..utilities.tiling import tile_image
from ..object_detection_models.object_detection_model import ObjectDetectionModel

# External Imports
import numpy as np


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_INTRAOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_checkbox_centroids.json"
)
FILEPATH_TO_PREOP_POSTOP_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "preop_postop_checkbox_centroids.json"
)
INTRAOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_INTRAOP_CENTROIDS, "r")
)
PREOP_POSTOP_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_PREOP_POSTOP_CENTROIDS, "r")
)


def extract_checkboxes(
    image: Image.Image,
    detection_model: ObjectDetectionModel,
    side: Literal["intraoperative", "preoperative"],
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float = 0.5,
    vertical_overlap_ratio: float = 0.5,
) -> Dict[str, str]:
    """Extracts checkbox data from an image of a chart.

    Args:
        `image` (Image.Image):
            The image to extract checkboxes from.
        `detection_model` (ObjectDetectionModel):
            An object that implements the ObjectDetectionModel interface.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Returns:
        A dictionary mapping the name of checkboxes to "checked" or "unchecked".
    """
    if side.lower() == "intraoperative":
        centroids = INTRAOP_CENTROIDS
    elif side.lower() == "preoperative":
        centroids = PREOP_POSTOP_CENTROIDS
    else:
        raise ValueError(
            f'Invalid selection for side. Must be one of ["intraoperative", "preoperative"], value supplied was {side}'
        )

    checkbox_bboxes: List[BoundingBox] = detect_checkboxes(
        image,
        detection_model,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    names: Dict[str, str] = find_checkbox_names(checkbox_bboxes, centroids, image.size)
    return names


def detect_checkboxes(
    image: Image.Image,
    detection_model: ObjectDetectionModel,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
) -> List[BoundingBox]:
    """Uses an object detector to detect checkboxes and their state on an image.

    Args:
        `image` (Image.Image):
            The image to extract checkboxes from.
        `detection_model` (ObjectDetectionModel):
            An object that implements the ObjectDetectionModel interface.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Returns:
        A list of Detection objects encoding the location and state of checkboxes.
    """
    image_tiles: List[List[Image.Image]] = tile_image(
        image,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    detections: List[List[List[Detection]]] = [
        [detection_model(tile, verbose=False)[0] for tile in row] for row in image_tiles
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
        threshold=0.8,
        overlap_comparator=intersection_over_minimum,
    )
    return [det.annotation for det in detections]


def find_checkbox_names(
    checkboxes: List[BoundingBox],
    centroids: Dict[str, Tuple[float, float]],
    imsize: Tuple[int, int],
    threshold: float = 0.025,
) -> Dict[str, str]:
    """Finds the names of checkboxes.

    The checkboxes must be the locations after image registration, or else they will
    be incorrect due to the naive nature of this solution, which is to merely find
    the closest checkbox based on the centroid of a sample of 10 ground-truth
    checkboxes.

    Args:
        `checkboxes` (List[BoundingBox]):
            The checkboxes to classify. The 'category' attribute should be 'checked'
            or 'unchecked'.
        `centroids` (Dict[str, Tuple[float, float]]):
            The centroids of a sample of 10 checkboxes.
        `threshold` (float):
            The threshold that determines how far a centroid can be before it
            is totally ruled out. If all checkbox centroids are more than the
            threshold away, there is no associated name for that checkbox.
            Defaults to 2.5% of the image's width and height.

    Returns:
        A dictionary that maps the name of the checkbox to 'checked' or 'unchecked'.
    """

    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Computes euclidean distance."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    checkbox_values: Dict[str, str] = dict()
    for ckbx in checkboxes:
        center = ckbx.center[0] / imsize[0], ckbx.center[1] / imsize[1]
        distance_to_all_centroids: Dict[str, float] = {
            name: distance(center, centroid) for (name, centroid) in centroids.items()
        }
        checkbox_too_far_from_any_centroid: bool = all(
            [dist > threshold for dist in list(distance_to_all_centroids.values())]
        )
        closest_checkbox_centroid: str = min(
            distance_to_all_centroids, key=distance_to_all_centroids.get
        )
        checkbox_values[closest_checkbox_centroid] = ckbx.category

    return checkbox_values


def find_interaoperative_checkbox_names(
    intraoperative_checkboxes: List[BoundingBox], threshold: float = 0.025
) -> Dict[str, str]:
    """Finds the names of intraoperative checkboxes."""
    return find_checkbox_names(intraoperative_checkboxes, INTRAOP_CENTROIDS, threshold)


def find_preoperative_checkbox_names(
    preoperative_checkboxes: List[BoundingBox], threshold: float = 0.025
) -> Dict[str, str]:
    """Finds the names of preoperative checkboxes."""
    return find_checkbox_names(
        preoperative_checkboxes, PREOP_POSTOP_CENTROIDS, threshold
    )
