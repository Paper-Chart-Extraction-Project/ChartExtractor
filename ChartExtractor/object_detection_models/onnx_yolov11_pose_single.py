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
from typing import Dict, List, Literal, Tuple

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


class OnnxYolov11PoseSingle(ObjectDetectionModel):
    """Provides a wrapper for a yolov11 pose ONNX model.
    
    This class inherits from the `ObjectDetectionModel` interface, enabling us to use the onnx
    model within our program through a consistent interface.

    Attributes:
        model:
            The underlying onnx runtime model.
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
        self.classes = self.load_classes(model_metadata_filepath)

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
        classes: Dict = read_yaml_file(model_metadata_filepath, potential_err_msg)
        return classes

    def __call__(
        self,
        images: List[np.array],
        confidence: float = 0.75,
        iou_threshold: float = 0.5
    ) -> List[List[Detection]]:
        """Runs the model on a list of images.

        Args:
            images (List[np.array]):
                A list of images read by cv2.imread.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.
            iou_threshold (float):
                The intersection over union threshold under which to remove
                detections via non-maximum suppression.

        Returns:
            A list of detections for each image.
        """
        if not isinstance(images, list):
            images = [images]
        detections: List[List[Detection]] = [
            self.detect(im, confidence, iou_threshold) for im in images
        ]
        return detections

    def detect(
        self,
        image: np.array,
        confidence: float,
        iou_threshold: float,
    ) -> List[Detection]:
        """Runs the model on a single image.

        Args:
            image (np.array):
                The image to detect on.
            confidence (float):
                The level of confidence below which all detections are culled.
                Default of 0.5.
            iou_threshold (float):
                The intersection over union threshold under which to remove
                detections via non-maximum suppression.

        Returns:
            A list of detections on the image.
        """
        image: np.array = self.preprocess_image(image)
        image: np.array = image.transpose((2, 0, 1))
        image: np.array = np.expand_dims(image, axis=0)
        pred_results = self.model.run(None, {"images": image})
        detections = self.postprocess_results(pred_results, confidence, iou_threshold)
        return [
            Detection(
                Keypoint(
                    Point(d[4].item(), d[5].item()),
                    BoundingBox(
                        self.classes[d[7]],
                        d[0].item(),
                        d[1].item(),
                        d[2].item(),
                        d[3].item(),
                    ),
                ),
                d[6].item()
            )
            for d in detections
        ]

    def preprocess_image(
        self,
        image: np.array,
        resize_method: Literal["resize", "letterbox"] = "resize",
    ) -> np.array:
        """Preprocesses an image for running in a yolov11 model.

        Args:
            image (np.array):
                An image read by cv2.imread().
            resize_method (Literal):
                Determines the method of resizing the image.
                One of ("resize", "letterbox").

        Returns:
            A preprocessed image.
        """
        if resize_method == "letterbox":
            image, _ = self.letterbox(image, (self.input_im_width, self.input_im_height))
        else:
            image: np.array = cv2.resize(
               image,
               (self.input_im_width, self.input_im_height),
               interpolation=cv2.INTER_LINEAR,
            )
        image: np.array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def postprocess_results(
        self,
        pred_results,
        confidence_threshold: float,
        iou_threshold: float,
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
            iou_threshold (float):
                The intersection over union threshold under which to remove
                detections via non-maximum suppression.

        Returns:
            A list of Detection objects.
        """
        pred_results = pred_results[0][0]

        confidences = pred_results[4, :]
        mask = confidences >= confidence_threshold  # Create a mask for cells

        if not np.any(mask):  # check if anything passes the confidence threshold
            return np.empty((0, 8))

        filtered_output = pred_results[:, mask]  # Apply mask to filter cells
        confidences = confidences[mask]  # filter confidences
        class_indices = np.argmax(filtered_output[4:5, :], axis=0)  # Get class indices

        x, y, w, h = filtered_output[:4, :]
        x1 = x - (w / 2)
        y1 = y - (h / 2)
        x2 = x + (w / 2)
        y2 = y + (h / 2)
        kpx = filtered_output[5, :]
        kpy = filtered_output[6, :]
        
        predictions = np.stack((x1, y1, x2, y2, kpx, kpy, confidences, class_indices), axis=1)
        predictions = predictions[self.keypoint_not_in_box(predictions)]
        predictions = predictions[self.non_max_suppression(predictions, iou_threshold)]
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
        self, predictions: np.ndarray, iou_threshold
    ) -> np.ndarray:
        """Performs non-maximum suppression on a list of predicted boxes.

        Args:
            predictions (np.ndarray):
                A numpy ndarray of bounding boxes in the format
                (x1, y1, x2, y2, kpx, kpy, conf, cls index).
            iou_threshold (float):
                The threshold above which to consider two boxes to be overlapping.

        Returns:
            A numpy array of booleans that encode which boxes need to be removed
            to perform non-maximum suppression. Use as a mask.
        """
        indexes_of_box: Tuple[int, int] = [0, 3]
        index_of_confidence: int = 6
        index_of_category: int = 7

        rows, _ = predictions.shape

        sort_index = np.flip(predictions[:, index_of_confidence].argsort())
        predictions = predictions[sort_index]

        boxes = predictions[:, indexes_of_box[0]:indexes_of_box[1]+1]
        categories = predictions[:, index_of_category]
        ious = self.batch_iou(boxes, boxes)
        ious = ious - np.eye(rows)

        keep = np.ones(rows, dtype=bool)

        for index, (iou, category) in enumerate(zip(ious, categories)):
            if not keep[index]:
                continue

            condition = (iou > iou_threshold) & (categories == category)
            keep = keep & ~condition
        
        return keep[sort_index.argsort()]
    
    def keypoint_not_in_box(self, predictions: np.ndarray) -> np.ndarray:
        """Generates a mask that can filter keypoints that aren't in the box.
        
        Args:
            predictions (np.ndarray):
                A numpy ndarray of bounding boxes in the format
                (x1, y1, x2, y2, kpx, kpy, conf, cls index).

        Returns:
            A mask showing which predictions have keypoints that are in bounds.
        """
        x1 = predictions[:, 0]
        y1 = predictions[:, 1]
        x2 = predictions[:, 2]
        y2 = predictions[:, 3]
        kpx = predictions[:, 4]
        kpy = predictions[:, 5]
        kpx_in_bounds = np.logical_and(x1<kpx, kpx<x2)
        kpy_in_bounds = np.logical_and(y1<kpy, kpy<y2)
        kp_in_bounds = np.logical_and(kpx_in_bounds, kpy_in_bounds)
        return kp_in_bounds
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        colors: List[Tuple] = None,
        mask_alpha: float = 0.3,
    ) -> np.ndarray:
        """Draws detections onto images.

        Args:
            image (np.ndarray):
                An image read by cv2.imread().
            detections (List[Detections]):
                The predictions as Detection objects.
            colors (List[Tuple]):
                The colors to use for the objects.
            mask_alpha (float):
                The alpha level for the bounding boxes.

        Returns:
            An image with detections drawn on top.
        """

        def get_color(category: int) -> Tuple[int, int, int]:
            if colors is None:
                return np.random.uniform(0, 255, size=(1, 3)).tolist()[0]
            else:
                return colors[category]

        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = image.copy()


        # Draw bounding boxes, masks, and text annotations
        for detection in detections:
            category = detection.annotation.category
            box = detection.annotation.box
            score = detection.confidence

            color = get_color(category)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Draw fill rectangle for mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare text (label and score)
            caption = f"{category} {int(score * 100)}%"

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
