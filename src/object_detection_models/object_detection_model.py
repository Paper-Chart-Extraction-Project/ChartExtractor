"""Defines an interface (ObjectDetectionModel) for any object detector to be used with the program.

The state of the art for object detection changes every few months, and it will be helpful to test
new or alternative object detectors in the future. Rather than write all new functionality in other
places, new detectors will simply need to create an subclass of ObjectDetectionModel, and so long
as they override the __call__ method to detect on a PIL image and output a list of Detections,
all other functions don't need changing.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import List
from utilities.detections import Detection
from PIL import Image


class ObjectDetectionModel(ABC):
    """Abstract base class for object detection models."""

    @abstractstaticmethod
    def from_weights_path(self, model_weights_path:Path):
        """Initializes the ObjectDetectionModel from a path to its weights.
        
        Args:
            model_path (Path):
                The path to the model's weights file.

        Raises:
            FileNotFoundError:
                If the filepath does not lead to a file.
        """
        pass

    @abstractstaticmethod
    def from_model(self, model):
        """Initializes the ObjectDetectionModel from a model object.

        Args:
            model: 
                An object from another package that performs object detection.
        """
        pass

    @abstractmethod
    def __call__(self, image:Image.Image) -> List[Detection]:
        """Detections objects on the image.

        Args:
            image (Image.Image):
                A PIL image that this model detects on.
        
        Returns:
            A list of Detection objects.
        """
        pass