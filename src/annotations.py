"""Module that contains dataclasses for storing image annotations."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BoundingBox:
    """ """

    category: str
    left: int
    top: int
    right: int
    bottom: int

    @staticmethod
    def from_yolo(
        yolo_line: str,
        image_width: int,
        image_height: int,
        int_to_category: Dict[int, str],
    ):
        pass

    @staticmethod
    def from_coco(coco_dict: Dict):
        pass

    @property
    def center(self):
        pass

    @property
    def box(self):
        pass
