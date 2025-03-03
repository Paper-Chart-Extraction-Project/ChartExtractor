"""This module defines the `ObjectDetectionModel` interface, which serves as a base class
for any object detection model to be used with this program.

The field of object detection is constantly evolving, with new and improved models
emerging frequently. This interface facilitates the integration and testing of such models
within the program. Instead of rewriting functionality for each new detector, developers can
create subclasses that inherit from `ObjectDetectionModel`. As long as these subclasses
override the `__call__` method to handle object detection on a PIL image and return a list
of `Detection` objects, existing code remains compatible.

This approach promotes modularity, flexibility, and future-proofing of the program.
"""

# Built-in Imports
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from typing import List

# Internal Imports
from ..utilities.detections import Detection


class ObjectDetectionModel(ABC):
    """Abstract base class for object detection models.

    This class defines the interface that all concrete object detection models must adhere to.
    """

    @staticmethod
    @abstractmethod
    def from_weights_path(model_weights_path: Path) -> "ObjectDetectionModel":
        """Initializes the ObjectDetectionModel from a path to its weights.

        Args:
            model_path (Path):
                The path to the model's weights file.

        Raises:
            FileNotFoundError:
                If the filepath does not lead to a file.

        Returns:
            ObjectDetectionModel:
                An instance of the concrete subclass initialized from the weights.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_model(model) -> "ObjectDetectionModel":
        """Initializes the ObjectDetectionModel from a model object.

        Args:
            model:
                An object from another package that performs object detection.

        Returns:
            ObjectDetectionModel: An instance of the concrete subclass initialized from the model.
        """
        pass

    @abstractmethod
    def __call__(self, image: Image.Image) -> List[Detection]:
        """Detects objects on the image.

        Args:
            image (Image.Image):
                A PIL image that this model detects on.

        Returns (List[Detection]):
            A list of Detection objects.
        """
        pass
