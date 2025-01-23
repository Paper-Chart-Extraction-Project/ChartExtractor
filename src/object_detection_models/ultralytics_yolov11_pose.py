"""This module provides a wrapper class for Ultralytics YOLOv11 pose models.

This class, `UltralyticsYOLOv11Pose`, inherits from the `ObjectDetectionModel` class
and provides methods for:

* Loading the model from a weights path or an existing model instance.
* Performing object detection and pose estimation on images.
* Converting model predictions to a list of `Detection` objects

This module leverages the `Ultralytics` library for the underlying object detection
model and the `utilities` module for data structures (`BoundingBox`, `Keypoint`, and
`Point`).
"""

from pathlib import Path
from typing import Dict, List
from PIL import Image
from ultralytics import YOLO
from utilities.detections import Detection
from utilities.annotations import BoundingBox, Keypoint, Point
from object_detection_models.object_detection_model import ObjectDetectionModel


class UltralyticsYOLOv11Pose(ObjectDetectionModel):
    """UltralyticsYOLOv11Pose Object Detection Model with Pose Estimation.

    Inherits from `ObjectDetectionModel`

    This class provides a wrapper for the Ultralytics YOLOv11 object detection model
    with pose estimation capabilities. It allows you to perform object detection and
    retrieve keypoint information for detected objects.

    The reason we use pose estimation is for getting pinpoint accuracy in blood
    pressure and heart rate detection. By placing a single keypoint on the tip/center
    of the bloop pressure/heart rate symbol, a model can be trained not only to
    place a bounding box around the object, but predict the exact point we are
    interested in.

    Attributes:
        `model`:
            The underlying Ultralytics object detection model for pose estimation.
    """

    def __init__(self, model):
        """Initializes the `UltralyticsYOLOv11Pose` object from a YOLO model.

        Args:
            `model`:
                An Ultralytics YOLOv11 Pose model instance.
        """
        self.model = model

    @staticmethod
    def from_weights_path(model_weights_path: Path) -> "UltralyticsYOLOv11Pose":
        """Loads a `UltralyticsYOLOv11Pose` model from a path to the model weights.

        Args:
            `model_weights_path` (Path):
                Path to the model weights file.

        Returns:
            An instance of the `UltralyticsYOLOv11Pose` class from the weights path.
        """
        return UltralyticsYOLOv11Pose(YOLO(model_weights_path))

    @staticmethod
    def from_model(model) -> "UltralyticsYOLOv11Pose":
        """Creates a `UltralyticsYOLOv11Pose` object from YOLOv11 pose model.

        Args:
            `model`:
                A Ultralytics YOLOv11 pose object detection model instance.

        Returns:
            An instance of the `UltralyticsYOLOv11Pose` class from an object from the
            Ultralytics library.
        """
        return UltralyticsYOLOv11Pose(model)

    def __call__(self, image: Image.Image, **kwargs) -> List[Detection]:
        """Performs object detection and pose estimation on an image.

        Args:
            `image` (Image.Image):
                The image to perform detection on.
            `kwargs`:
                Any argument that Ultralytics Yolo model will take. Mostly
                used for 'conf' and 'verbose'.

        Returns:
            List[Detection]:
                A list of `Detection` objects containing bounding boxes, keypoints,
                and confidence scores.
        """
        predictions = self.model(image, verbose=False, **kwargs)
        detections: List[Keypoint] = self.predictions_to_detections(predictions)
        return detections

    def predictions_to_detections(self, preds) -> List[Detection]:
        """Converts model predictions to a list of `Detection` objects.

        Internal method used for processing model output.

        Args:
            `preds`:
                The model predictions from the Ultralytics YOLOv11 object.

        Returns:
            List[Detection]:
                A list of `Detection` objects.
        """
        if len(preds[0].boxes.data.tolist()) == 0:
            return list()

        id_to_cat: Dict[int, str] = preds[0].names
        points: List[Point] = [
            Point(kp[0][0], kp[0][1]) for kp in preds[0].keypoints.data.tolist()
        ]
        bboxes: List[BoundingBox] = [
            BoundingBox(id_to_cat[box[5]], box[0], box[1], box[2], box[3])
            for box in preds[0].boxes.data.tolist()
        ]
        confidences: List[float] = [box[4] for box in preds[0].boxes.data.tolist()]
        detections: List[Detection] = [
            Detection(Keypoint(p, bb), conf)
            for (p, bb, conf) in list(zip(points, bboxes, confidences))
        ]
        return detections
