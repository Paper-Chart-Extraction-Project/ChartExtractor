""" """

from pathlib import Path
from typing import Callable, List
from PIL import Image
from ultralytics import YOLO
from utilities.detections import Detection
from object_detection_model import ObjectDetectionModel


class UltralyticsYOLOv8(ObjectDetectionModel):
    """ """

    def __init__(self, model):
        """Initializes the UltralyticsYOLOv8 object.
        
        Args:
            model:
                The Ultralytics class for the YOLOv8 model.
        """
        self.model = model

    @staticmethod
    def from_weights_path(self, weights_path:Path) -> "UltralyticsYOLOv8":
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
    def from_model(self, model) -> "UltralyticsYOLOv8":
        """Creates an UltralyticsYOLOv8 object from the Ultralytics model object.
        
        Args:
            model:
                The Ultralytics class for the YOLOv8 model.
        """
        return UltralyticsYOLOv8(model)

    def __call__(self, Image: Image.Image) -> List[Detection]:
        """ """
        pass