"""Utilities for detecting and associating meaning to handwritten digits."""

# Built-in imports
from functools import reduce
import json
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# External imports
import numpy as np

# Internal imports
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.annotations import BoundingBox, Keypoint
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import (
    intersection_over_minimum,
    non_maximum_suppression,
    untile_detections,
)
from ..utilities.image_conversion import pil_to_cv2
from ..utilities.tiling import tile_image


MAX_BOX_WIDTH, MAX_BOX_HEIGHT = (0.0174507, 0.0236938)


def average_with_nones(list_with_nones: List[Optional[float]]) -> float:
    """Calculates the average of a list of floats that may contain None values.

    Args:
        `list_with_nones` (List[Optional[float]]):
            A list of floats that may contain None values.

    Returns:
        The average of the non-None values in the list.

    Raises:
        ZeroDivisionError: If the input list contains only None values.
    """
    add_with_none = lambda acc, x: acc + x if x is not None else acc
    len_with_none = lambda l: len(list(filter(lambda x: x is not None, l)))
    return reduce(add_with_none, list_with_nones) / len_with_none(list_with_nones)


def combine_dictionaries(dictionaries: List[Dict]):
    """Combines a list of dictionaries into one.

    Args:
        `dictionaries` (List[Dict]):
            A list of dictionaries to combine.

    Returns:
        A single dictionary with the contents of all the dictionaries.
    """
    combined_dictionary: Dict = dict()
    for dictionary in dictionaries:
        combined_dictionary.update(dictionary)
    return combined_dictionary


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


def detect_objects_using_tiling(
    image: Image.Image,
    detection_model: ObjectDetectionModel,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
    minimum_confidence: float = 0.5,
    nms_threshold: float = 0.5,
    overlap_comparator: Callable[[Detection, Detection], float] = intersection_over_minimum,
    sorting_fn: Callable[[Detection], float] = lambda det: det.annotation.area * det.confidence,
) -> List[Detection]:
    """Detects objects, especially small ones, using image tiling.

    Splits an image up into smaller tiles, runs the model on each tile, then untiles the detections
    and performs non-maximum suppression on the result.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `detection_model` (ObjectDetectionModel):
            The detection model to use. Can be any object that implements the ObjectDetectionModel
            protocol.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
            (Ex: 0.2 results in 20% of a tile overlapping with the tile on the left and 20%
            overlapping on the right.)
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.
            (Ex: 0.2 results in 20% of a tile overlapping with the tile on the top and 20%
            overlapping on the bottom.)
        `minimum_confidence` (float):
            The minimum confidence level. Any detection with a confidence score below this will not
            be added to the returned detections. Defaults to 0.5.
        `nms_threshold` (float):
            The threshold above which nms registers a 'match', and deletes all but the first
            detection in a 'group'. A group is determined by the sorting_fn. Defaults to 0.5.
        `overlap_comparator` (float):
            The function that determines how much two detections overlap. Defaults to the
            intersection of the detections divided by the minimum of the two detection's areas.
            This default prevents partial detections from remaining inside the full detection.
        `sorting_fn` (Callable[[Detection], float]):
            The function that applies a 'score' to each detection to determine which has priority
            when NMS deletes detections. Only the detection with the highest score in a group
            remains. Defaults to the detection's confidence times its area.
    
    Returns:
        A list of detections showing objects on the image that the object detection model was
        trained to identify.
    """
    image_tiles: List[List[Image.Image]] = tile_image(
        image,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    detections: List[List[List[Detection]]] = [
        [detection_model(pil_to_cv2(tile), confidence=minimum_confidence)[0] for tile in row]
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
        threshold=nms_threshold,
        overlap_comparator=overlap_comparator,
        sorting_fn=sorting_fn,
    )
    return detections


def get_detection_by_name(
    detections: List[Detection], name: str
) -> Optional[Detection]:
    """Gets a detection from a list of Detection objects, or returns None.

    Will return the first detection that matches the name.

    Args:
        `detections` (List[Detection]):
            The detections to check.
        `name` (str):
            The name of the detection to find.

    Returns:
        The first detection that matches the name, or None if it cannot be found.
    """
    try:
        return list(filter(lambda d: d.annotation.category == name, detections))[0]
    except:
        return None


def label_studio_to_bboxes(
    path_to_json_data: Path,
    desired_im_width: int = 3300,
    desired_im_height: int = 2550,
) -> List[BoundingBox]:
    """
    Convert the json data from label studio to a list of BoundingBox objects
    Args:
        path_to_json_data (Path):
            Path to the json data from label studio
    Returns:
        List[BoundingBox]:
            List of BoundingBox objects
    """
    json_data: List[Dict] = json.loads(open(str(path_to_json_data)).read())
    return {
        sheet_data["data"]["image"].split("-")[-1]: [
            BoundingBox(
                category=label["value"]["rectanglelabels"][0],
                left=label["value"]["x"] / 100 * desired_im_width,
                top=label["value"]["y"] / 100 * desired_im_height,
                right=(label["value"]["x"] / 100 + label["value"]["width"] / 100)
                * desired_im_width,
                bottom=(label["value"]["y"] / 100 + label["value"]["height"] / 100)
                * desired_im_height,
            )
            for label in sheet_data["annotations"][0]["result"]
        ]
        for sheet_data in json_data
    }


def read_detections_from_json(
    filepath: Path,
    detection_type: Union[BoundingBox, Keypoint]
) -> List[Detection]:
    """Deserializes detections from a json file.
    
    Args:
        filepath (Path):
            The filepath to the json detections.
        detection_type (Union[BoundingBox, Keypoint]):
            The type of detection that has been serialized.
            Passed to Detection.from_dict directly.

    Returns:
        A list of Detection objects from the encoded data.
    """
    json_data: Dict[str, Any] = json.loads(open(str(filepath), 'r').read())
    if not isinstance(json_data, list):
        raise ValueError(f"Data at {filepath} is not a list of detections.")
    
    detections: List[Detection] = [
        Detection.from_dict(det_dict, detection_type)
        for det_dict in json_data
    ]
    return detections
    


def write_detections_to_json(
    filepath: Path,
    detections: List[Detection]
) -> bool:
    """Serializes detections to a json file.
    
    Args:
        filepath (Path):
            The filepath to store the json detections at.
        detections (List[Detection]):
            The detections to serialize and save.

    Returns:
        True if the writing was a success, False otherwise.
    """
    detections_as_dicts: List[Dict[str, Any]] = [detection.to_dict() for detection in detections]
    json_data: str = json.dumps(detections_as_dicts)
    try:
        with open(str(filepath), 'w') as f:
            f.write(json_data)
        return True
    except Exception as e:
        print(f"Writing detections to json generated the following error:\n{e}")
        return False
