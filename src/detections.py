"""This module defines the Detection class representing a single object detection result.

This class is used to store the output of an object detection model, including:

* The predicted location of the object, represented by either a BoundingBox or a Keypoint instance (depending on the model's output format).
* The confidence score assigned by the model to this detection (a float between 0.0 and 1.0).
"""

from dataclasses import dataclass
from typing import List, Union
from annotations import BoundingBox, Keypoint, Point


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


def yolov8_results_to_detections(results) -> List[Detection]:
    """Converts YOLOv8 model results to a list of Detection objects.

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
            not found). The specific error message will be printed.
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
            Detection(Keypoint(Point(*keypoints[ix][0]), d.annotation), d.confidence)
            for ix, d in enumerate(detections)
        ]
    except Exception as e:
        print(e)
    return detections
