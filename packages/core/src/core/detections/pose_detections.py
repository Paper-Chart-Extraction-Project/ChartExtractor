"""Contains the PoseDetections class."""

from core.detections import BoundingBoxWithKeypointsDetection
from pydantic import BaseModel, ConfigDict


class PoseDetections(BaseModel):
    """A collection of bounding box with keypoints detections coupled with class and keypoint names.

    Attributes:
        detections (list[BoundingBoxWithKeypointsDetection]):
            The detected bounding boxs with their respective keypoints.
        bounding_box_class_names (list[str]):
            An ordered list of the names of the classes. If zipped with any bounding box
            detections category_scores, it will produce a list containing the name of each class
            along with the confidence probability assigned to it by the model.
        keypoint_names (dict[str, list[str]]):
            A dictionary that maps the names of bounding boxes with keypoints, to an ordered list
            of the names of the keypoints for that category of bounding box.
    """

    model_config = ConfigDict(frozen=True)

    detections: list[BoundingBoxWithKeypointsDetection]
    bounding_box_class_names: list[str]
    keypoint_names: dict[str, list[str]]
