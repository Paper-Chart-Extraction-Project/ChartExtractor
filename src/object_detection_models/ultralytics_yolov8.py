"""This module implements the `UltralyticsYOLOv8` wrapper class

The `UltralyticsYOLOv8` class, which inherits from the `ObjectDetectionModel` interface,
provides a wrapper for the YOLOv8 object detection model from the Ultralytics library.
It enables you to use the YOLOv8 model within your program through the common interface defined
in `object_detection_model.py`.

Key functionalities include:
    - Provides a common interface for detections (via the __call__ method).
    - Loading the YOLOv8 model from a weights file path.
    - Performing object detection on an image using the YOLOv8 model.
    - Converting the YOLOv8 model's output to a list of Detection objects.

These `Detection` objects encapsulate details about detected objects, including bounding boxes,
confidence scores, and potentially keypoints (if available in the model's output).

This approach simplifies the integration and usage of YOLOv8 within this program, promoting code
modularity and reusability.
"""

from pathlib import Path
from typing import List
from PIL import Image
from ultralytics import YOLO
from utilities.detections import Detection
from utilities.annotations import BoundingBox, Keypoint, Point
from object_detection_models.object_detection_model import ObjectDetectionModel


class UltralyticsYOLOv8(ObjectDetectionModel):
    """Provides a wrapper for the YOLOv8 object detection model from the Ultralytics library.

    This class inherits from the `ObjectDetectionModel` interface, enabling us to use the YOLOv8
    model within our program through a consistent interface.

    Attributes:
        model:
            The underlying Ultralytics YOLOv8 model object.
    """

    def __init__(self, model):
        """Initializes the UltralyticsYOLOv8 object.

        Args:
            model:
                The Ultralytics class for the YOLOv8 model.
        """
        self.model = model

    @staticmethod
    def from_weights_path(weights_path: Path) -> "UltralyticsYOLOv8":
        """Creates an UltralyticsYOLOv8 object from a path to the weights file.

        Args:
            weights_path (Path):
                A path leading to the model's weights.pt file.

        Returns (UltralyticsYOLOv8):
            An UltralyticsYOLOv8 object.
        """
        model = YOLO(str(weights_path))
        return UltralyticsYOLOv8.from_model(model)

    @staticmethod
    def from_model(model) -> "UltralyticsYOLOv8":
        """Creates an UltralyticsYOLOv8 object from the Ultralytics model object.

        Args:
            model:
                The Ultralytics class for the YOLOv8 model.
        """
        return UltralyticsYOLOv8(model)

    def __call__(self, image: Image.Image, **kwargs) -> List[Detection]:
        """ """
        results = self.model(image, **kwargs)
        detections: List[Detection] = self.yolov8_results_to_detections(results)
        return detections

    def yolov8_results_to_detections(self, results) -> List[Detection]:
        """Converts ultralytics' YOLOv8 model object's results to a list of Detection objects.

        Args:
            results:
                List containing the output from a YOLOv8 model prediction. Refer to the YOLOv8
                documentation for details on the output format.

        Returns:
            A list of Detection objects. Each Detection object contains information about a
            detected object including its bounding box (category, coordinates), and confidence
            score. Additionally, if keypoints are present in the results, they are added
            to the Detection objects.

        Raises:
            Exception:
                If an error occurs during processing of the results (e.g., keypoints are
                not found).
        """
        detections: List[Detection] = [
            Detection(
                annotation=BoundingBox(
                    category=results[0].names[box_conf_cls[5]],
                    left=box_conf_cls[0],
                    top=box_conf_cls[1],
                    right=box_conf_cls[2],
                    bottom=box_conf_cls[3],
                ),
                confidence=box_conf_cls[4],
            )
            for box_conf_cls in results[0].boxes.data.tolist()
        ]
        try:
            keypoints = results[0].keypoints.data.tolist()
            detections = [
                Detection(
                    Keypoint(Point(*keypoints[ix][0]), d.annotation), d.confidence
                )
                for ix, d in enumerate(detections)
            ]
        except Exception:
            pass
        return detections
