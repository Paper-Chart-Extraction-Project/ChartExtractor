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
from typing import Dict, List, Tuple

# External imports
import cv2
import numpy as np
import onnxruntime as ort

# Internal imports
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import non_maximum_suppression
from ..utilities.read_config import read_yaml_file


class_num = 256
NUM_HEADS: int = 3
STRIDES: List[int] = [8, 16, 32]
MAPSIZE: List[List[int]] = [[80, 80], [40, 40], [20, 20]]
MESHGRID: List = list()
for index in range(NUM_HEADS):
    for ix in range(MAPSIZE[index][0]):
        for jx in range(MAPSIZE[index][1]):
            MESHGRID.append(jx + 0.5)
            MESHGRID.append(ix + 0.5)


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
    def load_classes(model_metadata_filepath: Path) -> Dict:
        """Loads the classes from a yaml file into a list.
        
        Args:
            model_metadata_filepath (Path):
                The path to the model metadata.
        
        Raises:
            Exception:
                Any exception relating to loading a file.

        Returns:
            A dictionary mapping the numerical id of a class to the class' name.
        """
        potential_err_msg = "An exception has occured while loading the classes "
        potential_err_msg += "yaml file. Ensure the model metadata filepath is "
        potential_err_msg += "correct and the model's yaml file is correctly formatted."
        classes: Dict = read_yaml_file(model_metadata_filepath)
        return classes

    def __call__(
        self, images: List[np.array], confidence: float = 0.5
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
        image: np.array = self.preprocess_image(image)
        image: np.array = image.transpose((2, 0, 1))
        image: np.array = np.expand_dims(image, axis=0)
        pred_results = self.model.run(None, {"images": image})
        detections = self.postprocess_results(pred_results, confidence)
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
        # image: np.array = cv2.resize(
        #    image,
        #    (self.input_im_width, self.input_im_height),
        #    interpolation=cv2.INTER_LINEAR,
        # )
        image, _ = self.letterbox(image, (self.input_im_width, self.input_im_height))
        image: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def postprocess_results(
        self,
        pred_results,
        confidence_threshold: float,
    ) -> List[Detection]:
        """Processes the raw results from the onnx model.

        I don't like this method. It was copy pasted from someone else's code and it is
        very obtuse and difficult to understand. Not only that but it is horribly slow.

        Args:
            pred_results:
                The raw predictions from the onnx model.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.

        Returns:
            A list of Detection objects.
        """
        pred_results = pred_results[0][0]
        pred_results[4:, :] = 1 / (1 + np.exp(-pred_results[4:, :]))

        confidences = np.max(pred_results[4:, :], axis=0)  # Get max confidence
        mask = confidences >= confidence_threshold  # Create a mask for cells

        if not np.any(mask):  # check if anything passes the confidence threshold
            return np.empty((0, 6))

        filtered_output = pred_results[:, mask]  # Apply mask to filter cells
        confidences = confidences[mask]  # filter confidences
        class_indices = np.argmax(filtered_output[4:, :], axis=0)  # Get class indices

        x, y, w, h = filtered_output[:4, :]
        x1 = np.round(x - w / 2).astype(np.int64)
        y1 = np.round(y - h / 2).astype(np.int64)
        x2 = np.round(x + w / 2).astype(np.int64)
        y2 = np.round(y + h / 2).astype(np.int64)

        predictions = np.stack((x1, y1, x2, y2, confidences, class_indices), axis=1)
        predictions = predictions[self.non_max_suppression(predictions)]
        return predictions

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        fill_value: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resizes and reshapes images while maintaining aspect ratio by adding padding.

        Args:
            image (np.ndarray):
                An image read by cv2.imread().
            new_shape (Tuple):
                The new shape to letterbox the image to.
            fill_value (Tuple):
                A tuple encoding the rgb value for the pad to use.

        Returns:
            A new image that is has been resized and padded on the top and bottom to
            match the new_shape.
        """
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value
        )

        return image, (top, left)

    def batch_iou(self, these_boxes: np.ndarray, those_boxes: np.ndarray) -> np.ndarray:
        """Computes the intersection over union between all pairs of boxes in two lists.

        Args:
            these_boxes (np.ndarray):
                A numpy ndarray of bounding boxes in the format (x1, y1, x2, y2, conf, cls index).
            those_boxes (np.ndarray):
                A numpy ndarray of bounding boxes in the format (x1, y1, x2, y2, conf, cls index).

        Returns:
            A matrix encoding the intersection over union between all pairs of boxes between
            the boxes in these_boxes and the boxes in those_boxes.
        """
        box_area = lambda box: (box[2] - box[0]) * (box[3] - box[1])

        these_areas = box_area(these_boxes.T)
        those_areas = box_area(those_boxes.T)

        top_left = np.maximum(these_boxes[:, None, :2], those_boxes[:, :2])
        bottom_right = np.minimum(these_boxes[:, None, 2:], those_boxes[:, 2:])

        intersection_areas = np.prod(
            np.clip(bottom_right - top_left, a_min=0, a_max=None), 2
        )

        return intersection_areas / (
            these_areas[:, None] + those_areas - intersection_areas
        )

    def non_max_suppression(
        self, predictions: np.ndarray, iou_threshold: float = 0.5
    ) -> np.ndarray:
        """Performs non-maximum suppression on a list of predicted boxes.

        Args:
            predictions (np.ndarray):
                A numpy ndarray of bounding boxes in the format (x1, y1, x2, y2, conf, cls index).
            iou_threshold (float):
                The threshold above which to consider two boxes to be overlapping.

        Returns:
            A numpy array of booleans that encode which boxes need to be removed
            to perform non-maximum suppression. Use as a mask.
        """
        rows, _ = predictions.shape

        sort_index = np.flip(predictions[:, 4].argsort())
        predictions = predictions[sort_index]

        boxes = predictions[:, :4]
        categories = predictions[:, 5]
        ious = self.batch_iou(boxes, boxes)
        ious = ious - np.eye(rows)

        keep = np.ones(rows, dtype=bool)

        for index, (iou, category) in enumerate(zip(ious, categories)):
            if not keep[index]:
                continue

            condition = (iou > iou_threshold) & (categories == category)
            keep = keep & ~condition

        return keep[sort_index.argsort()]

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        predictions: np.ndarray,
        classes: List[Tuple] = None,
        colors: List[Tuple] = None,
        mask_alpha: float = 0.3,
    ) -> np.ndarray:
        """Draws detections onto images.

        Args:
            image (np.ndarray):
                An image read by cv2.imread().
            predictions (np.ndarray):
                The predictions as a numpy ndarray.
            colors (List[Tuple]):
                The colors to use for the objects.
            mask_alpha (float):
                The alpha level for the bounding boxes.

        Returns:
            An image with detections drawn on top.
        """

        def get_color(class_id: int) -> Tuple[int, int, int]:
            if colors is None:
                return np.random.uniform(0, 255, size=(1, 3)).tolist()[0]
            else:
                return colors[class_id]

        def get_class_str(class_id: int) -> str:
            if classes is None:
                return str(class_id)
            else:
                return classes[class_id]

        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = image.copy()

        boxes, scores, class_ids = (
            predictions[:, :4].tolist(),
            predictions[:, 4].tolist(),
            predictions[:, 5].tolist(),
        )

        # Draw bounding boxes, masks, and text annotations
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = get_color(int(class_id))
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw fill rectangle for mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare text (label and score)
            label = get_class_str(int(class_id))
            caption = f"{label} {int(score * 100)}%"

            # Calculate text size and position
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size,
                thickness=text_thickness,
            )
            th = int(th * 1.2)

            # Draw filled rectangle for text background
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

            # Draw text over the filled rectangle
            cv2.putText(
                det_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        # Blend the mask image with the original image
        det_img = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

        return det_img
