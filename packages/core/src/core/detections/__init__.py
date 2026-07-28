from .bounding_box_detection import BoundingBoxDetection
from .keypoint_detection import KeypointDetection
from .bounding_box_with_keypoints_detection import BoundingBoxWithKeypointsDetection
from .object_detections import ObjectDetections
from .pose_detections import PoseDetections

__all__: list[str] = [
    "BoundingBoxDetection",
    "KeypointDetection",
    "BoundingBoxWithKeypointsDetection",
    "ObjectDetections",
    "PoseDetections",
]
