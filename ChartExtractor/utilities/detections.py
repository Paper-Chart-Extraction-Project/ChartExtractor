"""This module defines the Detection class representing a single object detection result.

This class is used to store the output of an object detection model, including:

* The predicted location of the object, represented by either a BoundingBox or a Keypoint instance (depending on the model's output format).
* The confidence score assigned by the model to this detection (a float between 0.0 and 1.0).
"""

# Built-in Imports
from dataclasses import dataclass
from typing import Any, Dict, Union

# Internal Imports
from ..utilities.annotations import BoundingBox, Keypoint


@dataclass
class Detection:
    """Represents a single detection result from an object detection model.

    Attributes:
        annotation:
            An instance of either BoundingBox or Keypoint class, depending on the
            type of annotation used for localization (bounding box or keypoints).
        confidence:
            A float value between 0.0 and 1.0 representing the confidence score
            assigned by the object detection model to this detection.
    """

    annotation: Union[BoundingBox, Keypoint]
    confidence: float
    
    @staticmethod
    def from_dict(detection_dict: Dict[str, Any], annotation_type: Union[BoundingBox, Keypoint]):
        """Creates a `Detection` from a dictionary of data.

        Args:
            detection_dict (Dict[str, Any]):
                The dictionary with the `Detection` data.
            annotation_type (Union[BoundingBox, Keypoint]):
                The type of annotation in the 

        Returns:
            A `Detection` object with the data from the dictionary.
        """
        annotation: Union[BoundingBox, Keypoint] = annotation_type.from_dict(
            detection_dict["annotation"]
        )
        confidence: float = detection_dict["confidence"]
        return Detection(annotation, confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Converts this detection to a dictionary."""
        return {
            "annotation": self.annotation.to_dict(),
            "confidence": self.confidence
        }
