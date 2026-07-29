from .types import AnnotationId
from .bounding_box import BoundingBox
from .keypoint import Keypoint, VisibilityStatus
from .bounding_box_with_keypoints import BoundingBoxWithKeypoints

type Annotation = BoundingBox | Keypoint | BoundingBoxWithKeypoints

__all__: list[str] = [
    "Annotation",
    "AnnotationId",
    "BoundingBox",
    "BoundingBoxWithKeypoints",
    "Keypoint",
    "VisibilityStatus",
]
