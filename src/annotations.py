"""Module that contains dataclasses for storing image annotations."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BoundingBox:
    """The 'BoundingBox' class represents a bounding box around an object in an image.


    Attributes :
        category (str) - The category of the object within the bounding box.
        left (int) - The x-coordinate of the top-left corner of the bounding box.
        top (int) - The y-coordinate of the top-left corner of the bounding box.
        right (int) - The x-coordinate of the bottom-right corner of the bounding box.
        bottom (int) - The y-coordinate of the bottom-right corner of the bounding box.


    Constructors :
        from_yolo(yolo_line: str, image_width: int, image_height: int, int_to_category: Dict[int, str])
            Constructs a 'BoundingBox' from a line in a YOLO formatted labels file. It requires the original image dimensions and a dictionary mapping category IDs to category names.

        from_coco(coco_annotation: Dict, image_metadata: Dict, categories: List[Dict])
            Constructs a 'BoundingBox' from an annotation in a COCO data JSON file. It requires the annotation dictionary, image metadata dictionary, and a list of category dictionaries.


    Properties :
        center (Tuple[int]) - A tuple containing the (x, y) coordinates of the bounding box's center.
        box (List[int]) - A list containing the bounding box coordinates as [left, top, right, bottom].
    """

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
    def center(self) -> Tuple[float]:
        """This BoundingBox's center."""
        return (
            self.left + (1 / 2) * (self.right - self.left),
            self.top + (1 / 2) * (self.bottom - self.top),
        )

    @property
    def box(self) -> List[int]:
        """A list containing this box's [left, top, right, bottom]."""
        return [self.left, self.top, self.right, self.bottom]


@dataclass
class Keypoint:
    """ """

    keypoint: Tuple[int]
    bounding_box: BoundingBox

    @staticmethod
    def from_yolo(
        yolo_line: str,
        image_width: int,
        image_height: int,
        int_to_category: Dict[int, str],
    ):
        """Constructs a keypoint from a line in a yolo formatted labels file.

        Because the yolo format stores data in normalized xywh format (from 0 to 1), this method
        requires the original image's width and height. The 'visible' data is optional, and is not
        read to create the object.

        Args :
            yolo_line (str) - A string in the yolo label format (c x y w h kpx kpy v).
            image_width (int) - The original image's width.
            image_height (int) - The original image's height.
            int_to_category (Dict) - A dictionary that maps the number in the label to the category.

        Returns : A BoundingBox object containing the yolo_line's data.
        """
        pass

    @property
    def category(self) -> str:
        """This keypoint's category."""
        return self.bounding_box.category

    @property
    def center(self) -> Tuple[float]:
        """This keypoint's boundingbox center."""
        return self.bounding_box.center

    @property
    def box(self) -> Tuple[float]:
        """This keypoints boundingbox's [left, top, right, bottom]."""
        return self.bounding_box.box
