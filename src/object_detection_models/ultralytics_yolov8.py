""" """

from pathlib import Path
from typing import List
from PIL import Image
from ultralytics import YOLO
from utilities.detections import Detection
from object_detection_model import ObjectDetectionModel


class UltralyticsYOLOv8(ObjectDetectionModel):
    """ """

    def __init__(self):
        """ """
        pass

    @staticmethod
    def from_weights_path(self, weights_path:Path) -> "UltralyticsYOLOv8":
        """ """
        pass

    @staticmethod
    def from_model(self, model) -> "UltralyticsYOLOv8":
        """ """
        pass

    def __call__(self, Image: Image.Image) -> List[Detection]:
        """ """
        pass