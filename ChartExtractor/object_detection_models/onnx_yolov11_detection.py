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
from typing import Dict, List

# External imports
import cv2
import numpy as np
import onnxruntime as ort
import yaml

# Internal imports
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import non_maximum_suppression


class_num = 256
NUM_HEADS: int = 3
STRIDES: List[int] = [8, 16, 32]
MAPSIZE: List[List[int]] = [[80, 80], [40, 40], [20, 20]]
MESHGRID: List = list()
for index in range(NUM_HEADS):
    for ix in range(MAPSIZE[index][0]):
        for jx in range(MAPSIZE[index][1]):
            MESHGRID.append(jx+0.5)
            MESHGRID.append(ix+0.5)


class OnnxYolov11Detection(ObjectDetectionModel):
    """Provides a wrapper for a yolov11 ONNX model.

    This class inherits from the `ObjectDetectionModel` interface, enabling us to use the onnx
    model within our program through a consistent interface.

    Attributes:
        model:
            The underlying onnx model.
    """

    def __init__(
        self,
        model_weights_filepath: Path,
        model_metadata_filepath: Path,
        input_im_width: int = 640,
        input_im_height: int = 640,
    ):
        """Initializes the onnx model.
        
        Args:
            model_weights_filepath (Path):
                The filepath to the model's weights.
            model_metadata_filepath (Path):
                The filepath to the metadata (for class names).
            input_im_width (int):
                The image width that the model accepts.
                Defaults to 640.
            input_im_height (int):
                The image height that the model accepts.
                Defaults to 640.
        """
        self.model = ort.InferenceSession(model_weights_filepath)
        self.input_im_width = input_im_width
        self.input_im_height = input_im_height
        self.classes = OnnxYolov11Detection.load_classes(model_metadata_filepath)
        
    def from_model(self):
        pass
    
    def from_weights_path(self):
        pass
    
    @staticmethod
    def load_classes(model_metadata_filepath) -> Dict[int, str]:
        """ """
        pass

    def __call__(
        self, 
        images: List[np.array],
        confidence: float = 0.5
    ) -> List[List[Detection]]:
        """Runs the model on a list of images.

        Args:
            images (List[np.array]):
                A list of images read by cv2.imread.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.

        Returns:
            A list of detections for each image.
        """
        if not isinstance(images, list):
            images = [images]
        detections: List[List[Detection]] = [
            self.detect(im, confidence) for im in images
        ]
        return detections

    def detect(
        self,
        image: np.array,
        confidence: float,
    ) -> List[Detection]:
        """Runs the model on a single image.
        
        Args:
            image (np.array):
                The image to detect on.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.

        Returns:
            A list of detections on the image.
        """
        im_width, im_height = image.shape[:2]
        image: np.array = self.preprocess_image(image)
        image: np.array = image.transpose((2, 0, 1))
        image: np.array = np.expand_dims(image, axis=0)
        pred_results = self.model.run(None, {'data': image})
        detections = self.postprocess_results(pred_results, im_width, im_height, confidence)
        return detections

    def preprocess_image(
        self,
        image: np.array,
    ) -> np.array:
        """Preprocesses an image for running in a yolov11 model.
        
        Args:
            image (np.array):
                An image read by cv2.imread().

        Returns:
            A preprocessed image.
        """
        image: np.array = cv2.resize(
            image,
            (self.input_im_width, self.input_im_height),
            interpolation=cv2.INTER_LINEAR
        )
        image: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image
    
    def postprocess_results(
        self,
        pred_results,
        image_width: int,
        image_height: int,
        confidence: float,
    ) -> List[Detection]:
        """Processes the raw results from the onnx model.
        
        I don't like this method. It was copy pasted from someone else's code and it is
        very obtuse and difficult to understand. Not only that but it is horribly slow.

        Args:
            pred_results:
                The raw predictions from the onnx model.
            image_width (int):
                The original width of the image.
            image_height (int):
                The original height of the image.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.

        Returns:
            A list of Detection objects.
        """
        def sigmoid(x) -> int:
            """Applies the sigmoid function to x.
            
            Args:
                x: the input number.

            Returns: 1/(1+e^-x)
            """
            return 1/(1+np.exp(-x))
        
        pred_results = [1/(1+np.exp(-pr)) for pr in pred_results]
        detections: List[Detection] = list()
        pred_results = [pred_results[i].reshape(-1) for i in range(len(pred_results))]
        scalar_w: float = image_width/self.input_im_width
        scalar_h: float = image_height/self.input_im_height
        grid_index: int = -2 # magic number...

        for index in range(NUM_HEADS):
            regression = pred_results[index * 2 + 0]
            classification = pred_results[index * 2 + 1]

            for h in range(MAPSIZE[index][0]):
                for w in range(MAPSIZE[index][1]):
                    grid_index += 2

                    if class_num == 1:
                        cls_max = classification[h*MAPSIZE[index][1]+w]
                        cls_index: int = 0
                    else:
                        cls_max, cls_index = max(
                            [
                                (classification[cl*MAPSIZE[index][1]+h*MAPSIZE[index][1]+w], cl)
                                for cl in range(class_num)
                            ]
                        )

                    if cls_max > confidence:
                        regdfl = list()
                        for lc in range(4): #four???
                            locval: int = 0
                            sfsum = np.sum(np.exp(
                                regression[((lc*16)+df)*MAPSIZE[index][0]*MAPSIZE[index][1]+h*MAPSIZE[index][1]+w])
                                for df in range (16)
                            )
                            for df in range(16):
                                sfval = np.exp(regression[((lc*16)+df) * MAPSIZE[index][0] * MAPSIZE[index][1] + h * MAPSIZE[index][1]+w]) / sfsum
                                locval += sfval * df
                            regdfl.append(locval)

                        x1 = (MESHGRID[grid_index+0] - regdfl[0]) * STRIDES[index]
                        y1 = (MESHGRID[grid_index+1] - regdfl[1]) * STRIDES[index]
                        x2 = (MESHGRID[grid_index+0] + regdfl[2]) * STRIDES[index]
                        y2 = (MESHGRID[grid_index+1] + regdfl[3]) * STRIDES[index]

                        xmin = max(0, x1*scalar_w)
                        ymin = max(0, y1*scalar_h)
                        xmax = min(image_width, x2*scalar_w)
                        ymax = min(image_height, y2*scalar_h)
                        
                        det = Detection(
                            BoundingBox(cls_index, xmin, ymin, xmax, ymax),
                            cls_max
                        )
                        detections.append(det)
        
        detections: List[Detection] = non_maximum_suppression(detections)
        return detections

