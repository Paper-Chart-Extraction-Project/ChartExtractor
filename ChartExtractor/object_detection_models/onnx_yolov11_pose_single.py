"""This module implements the `OnnxYolov11PoseSingle` wrapper class.

The `OnnxYolov11PoseSingle` class, which inherits the `ObjectDetectionModel` interface,
provides a wrapper for the YOLOv11 pose model using the onnx runtime. The poses that it
estimates are single keypoint poses, like the kind used in this project.

Key functionalities include:
    - Provides a common interface for detections (via the __call__ method).
    - Loading the YOLOv11 model from a weights file path.
    - Preprocessing images and postprocessing detections.
    - Performing pose estimation on an image using the YOLOv11 model.
    - Converting the YOLOv11 model's output to a list of Detection objects.

These `Detection` objects encapsulate details about detected objects, including bounding boxes,
confidence scores, and potentially keypoints (if available in the model's output).

This approach simplifies the integration and usage of YOLO within this program, promoting code
modularity and reusability.
"""

# Built-in imports
from pathlib import Path
from typing import Dict, List, Tuple

# External imports
import cv2
import numpy as np
import onnxruntime as ort

# Internal imports
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.annotations import BoundingBox, Keypoint, Point
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import non_maximum_suppression
from ..utilities.read_config import read_yaml_file


