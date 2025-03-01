"""This module implements the `OnnxYolov11Detection` wrapper class

The `OnnxYolov11Detection` class, which inherits from the `ObjectDetectionModel` interface,
provides a wrapper for the YOLOv11 object detection model using the onnx runtime.

Key functionalities include:
    - Provides a common interface for detections (via the __call__ method).
    - Loading the YOLOv11 model from a weights file path.
    - Preprocessing images and postprocessing detections.
    - Performing object detection on an image using the YOLOv11 model.
    - Converting the YOLOv11 model's output to a list of Detection objects.

These `Detection` objects encapsulate details about detected objects, including bounding boxes,
confidence scores, and potentially keypoints (if available in the model's output).

This approach simplifies the integration and usage of YOLOv8 within this program, promoting code
modularity and reusability.
"""

# Built-in imports
from pathlib import Path
from typing import List

# External imports
import numpy as np
import onnxruntime as ort

# Internal imports
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.detections import Detection


NUM_CLASSES: int = len(CLASSES)
NUM_HEADS: int = 3
STRIDES: List[int] = [8, 16, 32]
MAPSIZE: List[List[int]] = [[80, 80], [40, 40], [20, 20]]
MESHGRID: List = list()
for index in range(NUM_HEADS):
    for ix in range(MAPSIZE[index][0]):
        for jx in range(MAPSIZE[index][1]):
            MESHGRID.append(j+0.5)
            MESHGRID.append(i+0.5)


class OnnxYolov11Detection(ObjectDetectionModel):
    """Provides a wrapper for a yolov11 ONNX model.

    This class inherits from the `ObjectDetectionModel` interface, enabling us to use the onnx
    model within our program through a consistent interface.

    Attributes:
        model:
            The underlying onnx model.
    """

    def __init__(self, model_weights_filepath: Path):
        """Initializes the onnx model.
        
        Args:
            model_weights_filepath (Path):
                The filepath to the model's weights.
        """
        self.model = ort.InferenceSession(self.model_weights_filepath)


    def sigmoid(self, x) -> int:
        """Applies the sigmoid function to x.
        
        Args:
            x: the input number.

        Returns: 1/(1+e^-x)
        """
        return 1/(1+np.exp(-x))

    def preprocess_image(
            self,
            image: np.array,
            new_width: int,
            new_height: int
        ) -> np.array:
        """Preprocesses an image for running in a yolov11 model.
        
        Args:
            image (np.array):
                An image read by cv2.imread().
            new_width (int):
                The width to resize the image to.
            new_height (int):
                The height to resize the image to.

        Returns:
            A preprocessed image.
        """
        image: np.array = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        image: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def __call__(self, images: List[np.array]) -> List[List[Detection]]:
        """Runs the model on a list of images.

        Args:
            images (List[np.array]):
                A list of images read by cv2.imread.

        Returns:
            A list of detections for each image.
        """
        if not isinstance(images, list):
            images = [images]
        detections: List[List[Detection]] = [
            self.detect(im) for im in images
        ]
        return detections

    def detect(self, image: np.array) -> List[Detection]:
        """Runs the model on a single image.
        
        Args:
            image (np.array):
                The image to detect on.

        Returns:
            A list of detections on the image.
        """
        pass
