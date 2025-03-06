"""This module defines the `ObjectDetectionModel` protocol, which serves as a structural subtype
for any object detection model to be used with this program.

This protocol will help us define core functionality between objects that adhere to this
protocol without relying on inheritance.

Implementers of this protocol should explicitly subclass it to enable static checking.

This approach promotes modularity, flexibility, and future-proofing of the program.
"""

# Built-in Imports
from pathlib import Path
from PIL import Image
from typing import List, Protocol

# Internal Imports
from ..utilities.detections import Detection


class ObjectDetectionModel(Protocol):
    """Abstract base class for object detection models.

    This class defines the interface that all concrete object detection models must adhere to.
    """

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
