"""Module that contains dataclasses for storing image annotations."""

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple


Point = namedtuple("Point", ["x", "y"])


@dataclass
class BoundingBox:
    """The `BoundingBox` class represents a bounding box around an object in an image.


    Attributes :
        `category` (str) - The category of the object within the bounding box.
        `left` (float) - The x-coordinate of the top-left corner of the bounding box.
        `top` (float) - The y-coordinate of the top-left corner of the bounding box.
        `right` (float) - The x-coordinate of the bottom-right corner of the bounding box.
        `bottom` (float) - The y-coordinate of the bottom-right corner of the bounding box.


    Constructors :
        `from_yolo(yolo_line: str, image_width: int, image_height: int, int_to_category: Dict[int, str])`:
            Constructs a `BoundingBox` from a line in a YOLO formatted labels file. It requires the original image dimensions and a dictionary mapping category IDs to category names.

        `from_coco(coco_annotation: Dict, categories: List[Dict])`:
            Constructs a `BoundingBox` from an annotation in a COCO data JSON file. It requires the annotation dictionary and a list of category dictionaries.


    Properties :
        `center` (Tuple[int]) - A tuple containing the (x, y) coordinates of the bounding box's center.
        `box` (List[int]) - A list containing the bounding box coordinates as [left, top, right, bottom].


    Methods :
        `to_yolo(image_width: int, image_height: int, category_to_int: Dict[str, int])`:
            Writes a yolo formatted string using this bounding box's data.
    """

    category: str
    left: float
    top: float
    right: float
    bottom: float

    @staticmethod
    def from_yolo(
        yolo_line: str,
        image_width: int,
        image_height: int,
        id_to_category: Dict[int, str],
    ):
        """Constructs a `BoundingBox` from a line in a yolo formatted labels file.

        Because the yolo format stores data in normalized xywh format (from 0 to 1), this method
        requires the original image's width and height.

        Args :
            `yolo_line` (str) - A string in the yolo label format (c x y w h).
            `image_width` (int) - The original image's width.
            `image_height` (int) - The original image's height.
            `id_to_category` (Dict) - A dictionary that maps the number id in the label to the category.

        Returns : A `BoundingBox` object containing the yolo_line's data.
        """
        data = yolo_line.split()
        x, y, w, h = float(data[1]), float(data[2]), float(data[3]), float(data[4])
        x, y, w, h = (
            x * image_width,
            y * image_height,
            w * image_width,
            h * image_width,
        )
        left, top, right, bottom = (
            x - (1 / 2) * w,
            y - (1 / 2) * h,
            x + (1 / 2) * w,
            y + (1 / 2) * h,
        )
        category = id_to_category[int(data[0])]
        return BoundingBox(category, left, top, right, bottom)

    @staticmethod
    def from_coco(coco_annotation: Dict, categories: List[Dict]):
        """Constructs a `BoundingBox` from an annotation in a coco data json file.

        Args :
            `coco_annotation` (Dict) - A bounding box annotation from the 'annotations' section.
            `categories` (List[Dict]) - A list of dictionaries containing their numeric ids and categories.

        Returns : A `BoundingBox` object containing the coco annotation's data.
        """
        left, top, w, h = coco_annotation["bbox"]
        right, bottom = left + w, top + h
        category = list(
            filter(lambda c: c["id"] == coco_annotation["category_id"], categories)
        )[0]["name"]
        return BoundingBox(category, left, top, right, bottom)

    @property
    def center(self) -> Tuple[float]:
        """This `BoundingBox`'s center."""
        return (
            self.left + (1 / 2) * (self.right - self.left),
            self.top + (1 / 2) * (self.bottom - self.top),
        )

    @property
    def box(self) -> List[int]:
        """A list containing this `BoundingBox`'s [left, top, right, bottom]."""
        return [self.left, self.top, self.right, self.bottom]

    def to_yolo(
        self, image_width: int, image_height: int, category_to_id: Dict[str, int]
    ) -> str:
        """Writes the data from this `BoundingBox` into a yolo formatted string.

        Args :
            `image_width` (int) - The image's width that this boundingbox belongs to.
            `image_height` (int) - The image's height that this boundingbox belongs to.
            `category_to_id` (Dict[str, int]) - A dictionary that maps the category string to an id (integer).

        Returns : A string that encodes this `BoundingBox`'s data for a single line in a yolo label file.
        """
        c = category_to_id[self.category]
        x, y = self.center
        x /= image_width
        y /= image_height
        w = (self.right - self.left) / image_width
        h = (self.bottom - self.top) / image_height
        return f"{c} {x} {y} {w} {h}"


@dataclass
class Keypoint:
    """The `Keypoint` class represents a keypoint associated with an object in an image.

    Attributes :
        `keypoint` (Tuple[float]) - A tuple containing the (x, y) coordinates of the keypoint relative to the top-left corner of the image.
        `bounding_box` (BoundingBox) - A `BoundingBox` object that defines the bounding box around the object containing the keypoint.


    Constructors :
        `from_yolo(yolo_line: str, image_width: int, image_height: int, id_to_category: Dict[int, str])`:
            Constructs a Keypoint from a line in a YOLO formatted labels file. It requires the original image dimensions and a dictionary mapping category IDs to category names.
            **Note:** This method ignores the "visibility" information (denoted by 'v') in the YOLO format.


    Properties :
        `category` (str) - The category of the object the keypoint belongs to (inherited from the `bounding_box`).
        `center` (Tuple[float]) - The (x, y) coordinates of the bounding box's center (inherited from the `bounding_box`).
        `box` (Tuple[float]) - A list containing the bounding box coordinates as [left, top, right, bottom] (inherited from the `bounding_box`).


    Methods :
        `to_yolo(self, image_width: int, image_height: int, category_to_id: Dict[str, int]) -> str`:
            Generates a YOLO formatted string representation of this `Keypoint` object. It requires the image dimensions and a dictionary mapping category strings to integer labels.
    """

    keypoint: Point
    bounding_box: BoundingBox

    @staticmethod
    def from_yolo(
        yolo_line: str,
        image_width: int,
        image_height: int,
        id_to_category: Dict[int, str],
    ):
        """Constructs a `Keypoint` from a line in a yolo formatted labels file.

        Because the yolo format stores data in normalized xywh format (from 0 to 1), this method
        requires the original image's width and height. The 'visible' data is optional, and is not
        read to create the object.

        Args :
            `yolo_line` (str) - A string in the yolo label format (c x y w h kpx kpy v).
            `image_width` (int) - The original image's width.
            `image_height` (int) - The original image's height.
            `id_to_category` (Dict) - A dictionary that maps the id number in the label to the category.

        Returns : A `BoundingBox` object containing the yolo_line's data.
        """
        bounding_box = BoundingBox.from_yolo(
            yolo_line, image_width, image_height, id_to_category
        )
        keypoint_x = float(yolo_line.split()[5])
        keypoint_y = float(yolo_line.split()[6])
        keypoint = Point(keypoint_x * image_width, keypoint_y * image_height)
        return Keypoint(keypoint, bounding_box)

    @property
    def category(self) -> str:
        """This `Keypoint`'s category."""
        return self.bounding_box.category

    @property
    def center(self) -> Tuple[float]:
        """This `Keypoint`'s boundingbox center."""
        return self.bounding_box.center

    @property
    def box(self) -> Tuple[float]:
        """This keypoints boundingbox's [left, top, right, bottom]."""
        return self.bounding_box.box

    def to_yolo(
        self, image_width: int, image_height: int, category_to_id: Dict[str, int]
    ) -> str:
        """Writes the data from this `Keypoint` into a yolo formatted string.

        Args :
            `image_width` (int) - The image's width that this keypoint belongs to.
            `image_height` (int) - The image's height that this keypoint belongs to.
            `category_to_id` (Dict[str, int]) - A dictionary that maps the category string to an id (int).

        Returns : A string that encodes this `Keypoint`'s data for a single line in a yolo label file.
        """
        yolo_line = self.bounding_box.to_yolo(image_width, image_height, category_to_id)
        keypoint_x, keypoint_y = (
            self.keypoint.x / image_width,
            self.keypoint.y / image_height,
        )
        yolo_line += f" {keypoint_x} {keypoint_y}"
        return yolo_line
