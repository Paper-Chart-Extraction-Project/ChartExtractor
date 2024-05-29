"""Module that contains dataclasses for storing image annotations."""

from dataclasses import dataclass
from typing import Dict, List


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
        """Constructs a bounding box from a line in a yolo formatted labels file.

        Because the yolo format stores data in normalized xywh format (from 0 to 1), this method
        requires the original image's width and height.

        Args :
            yolo_line (str) - A string in the yolo label format (c x y w h).
            image_width (int) - The original image's width.
            image_height (int) - The original image's height.
            int_to_category (Dict) - A dictionary that maps the number in the label to the category.

        Returns : A BoundingBox object containing the yolo_line's data.
        """
        pass

    @staticmethod
    def from_coco(coco_annotation: Dict, image_metadata: Dict, categories: List[Dict]):
        """Constructs a bounding box from an annotation in a coco data json file.

        Args :
            coco_annotation (Dict) - A bounding box annotation from the 'annotations' section.
            image_metadata (Dict) - A dictionary from the 'images' section.
            categories (List[Dict]) - A list of dictionaries containing their numeric ids and categories.

        Returns : A BoundingBox object containing the coco annotation's data.
        """
        pass

    @property
    def center(self):
        pass

    @property
    def box(self):
        pass
