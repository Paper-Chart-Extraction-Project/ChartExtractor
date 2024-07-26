"""This module defines classes for representing bounding boxes and keypoints associated with objects in images. 

It also provides helper functions for constructing these objects from YOLO formatted labels.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings


class Point:
    """The `Point` class is a struct which contains an x and y value for a point.
    
    Attributes :
        `x` (float):
            The x coordinate for the point.
        `y` (float):
            The y coordinate for the point.
    """

    def __init__(self, x:float, y:float):
        """inits this point."""
        self.x = x
        self.y = y

    def __eq__(self, other):
        """Determines if two points are the same."""
        return self.x == other.x and self.y == other.y


@dataclass
class BoundingBox:
    """The `BoundingBox` class represents a bounding box around an object in an image.


    Attributes :
        `category` (str):
            The category of the object within the bounding box.
        `left` (float):
            The x-coordinate of the top-left corner of the bounding box.
        `top` (float):
            The y-coordinate of the top-left corner of the bounding box.
        `right` (float):
            The x-coordinate of the bottom-right corner of the bounding box.
        `bottom` (float):
            The y-coordinate of the bottom-right corner of the bounding box.


    Constructors :
        `from_yolo(yolo_line: str, image_width: int, image_height: int, int_to_category: Dict[int, str])`:
            Constructs a `BoundingBox` from a line in a YOLO formatted labels file. It requires the original image dimensions and a dictionary mapping category IDs to category names.

        `from_coco(coco_annotation: Dict, categories: List[Dict])`:
            Constructs a `BoundingBox` from an annotation in a COCO data JSON file. It requires the annotation dictionary and a list of category dictionaries.


    Properties :
        `center` (Tuple[int]):
            A tuple containing the (x, y) coordinates of the bounding box's center.
        `box` (List[int]):
            A list containing the bounding box coordinates as [left, top, right, bottom].


    Methods :
        `to_yolo(image_width: int, image_height: int, category_to_int: Dict[str, int]) -> str`:
            Writes a yolo formatted string using this bounding box's data.
        `validate_box_values(cls, left: float, top: float, right: float, bottom: float) -> None`:
            Validates the box parameters and throws a value error if left > right or top > bottom.
            Also issues a warning for the case when left == right or top == bottom letting the user
            know that they are constructing a degenerate rectangle.
    """

    category: str
    left: float
    top: float
    right: float
    bottom: float

    def __init__(
        self, category: str, left: float, top: float, right: float, bottom: float
    ):
        """Overrides the default constructor from dataclass to validate the parameters before constructing."""
        BoundingBox.validate_box_values(left, top, right, bottom)
        self.category = category
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

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
            `yolo_line` (str):
                A string in the yolo label format (c x y w h).
            `image_width` (int):
                The original image's width.
            `image_height` (int):
                The original image's height.
            `id_to_category` (Dict):
                A dictionary that maps the number id in the label to the category.

        Returns:
            A `BoundingBox` object containing the yolo_line's data.
        """
        data = yolo_line.split()
        x, y, w, h = float(data[1]), float(data[2]), float(data[3]), float(data[4])
        x, y, w, h = (
            x * image_width,
            y * image_height,
            w * image_width,
            h * image_height,
        )
        left, top, right, bottom = (
            x - (1 / 2) * w,
            y - (1 / 2) * h,
            x + (1 / 2) * w,
            y + (1 / 2) * h,
        )
        category = id_to_category.get(int(data[0]))
        if category == None:
            raise ValueError(
                f"Category {int(data[0])} not found in the id_to_category dictionary."
            )
        return BoundingBox(category, left, top, right, bottom)

    @staticmethod
    def from_coco(coco_annotation: Dict, categories: List[Dict]):
        """Constructs a `BoundingBox` from an annotation in a coco data json file.

        Args :
            `coco_annotation` (Dict): A bounding box annotation from the 'annotations' section.
            `categories` (List[Dict]): A list of dictionaries containing their numeric ids and categories.

        Returns:
            A `BoundingBox` object containing the coco annotation's data.
        """
        left, top, w, h = coco_annotation["bbox"]
        right, bottom = left + w, top + h
        try:
            category = list(
                filter(lambda c: c["id"] == coco_annotation["category_id"], categories)
            )[0].get("name")
        except IndexError:
            raise ValueError(
                f"Category {int(coco_annotation['category_id'])} not found in the categories list."
            )
        return BoundingBox(category, left, top, right, bottom)

    @classmethod
    def validate_box_values(
        cls, left: float, top: float, right: float, bottom: float
    ) -> None:
        """Validates the coordinates of a rectangle (bounding box).

        This classmethod ensures that the left coordinate is less than the right coordinate, and
        the top coordinate is less than the bottom coordinate. It raises a `ValueError` if these
        conditions are not met, indicating an invalid box configuration. If the left coordinate
        is equal to the right coordinate or if the top coordinate is equal to the bottom
        coordinate, this method issues a warning.

        Args:
            `left` (float):
                The left x-coordinate of the box.
            `top` (float):
                The top y-coordinate of the box.
            `right` (float):
                The right x-coordinate of the box.
            `bottom` (float):
                The bottom y-coordinate of the box.

        Raises:
            ValueError: If `left > right` or `top > bottom`.
        """
        if left > right:
            raise ValueError(
                f"Box's left side greater than its right side (Left:{left} > Right:{right})."
            )
        if top > bottom:
            raise ValueError(
                f"Box's top side greater than its bottom side (Top:{top} > Bottom:{bottom})."
            )
        if left == right and bottom == top:
            warnings.warn(
                f"Degenerate rectangle detected. All of the box's parameters are equal (Left:{left}, Top:{top}, Right:{right}, Bottom:{bottom}).",
                UserWarning,
            )
        elif left == right:
            warnings.warn(
                f"Degenerate rectangle detected. The box's left side equals its right side (Left:{left}, Top:{top}, Right:{right}, Bottom:{bottom}).",
                UserWarning,
            )
        elif top == bottom:
            warnings.warn(
                f"Degenerate rectangle detected. The box's top side equals its bottom side (Left:{left}, Top:{top}, Right:{right}, Bottom:{bottom}).",
                UserWarning,
            )

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

    def set_box(self, new_left: int, new_top: int, new_right: int, new_bottom: int):
        """Sets this BoundingBox's values for left, top, right, bottom.

        Args :
            new_left (int):
                The new left side for the box.
            new_top (int):
                The new top side for the box.
            new_right (int):
                The new right side for the box.
            new_bottom (int):
                The new bottom side for the box.
        """
        self.validate_box_values(new_left, new_top, new_right, new_bottom)
        return BoundingBox(
            category=self.category,
            left=new_left,
            top=new_top,
            right=new_right,
            bottom=new_bottom,
        )

    def to_yolo(
        self, image_width: int, image_height: int, category_to_id: Dict[str, int]
    ) -> str:
        """Writes the data from this `BoundingBox` into a yolo formatted string.

        Args :
            `image_width` (int):
                The image's width that this boundingbox belongs to.
            `image_height` (int):
                The image's height that this boundingbox belongs to.
            `category_to_id` (Dict[str, int]):
                A dictionary that maps the category string to an id (integer).

        Returns:
            A string that encodes this `BoundingBox`'s data for a single line in a yolo label file.
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
        `keypoint` (Tuple[float]):
            A tuple containing the (x, y) coordinates of the keypoint relative to the top-left corner of the image.
        `bounding_box` (BoundingBox):
            A `BoundingBox` object that defines the bounding box around the object containing the keypoint.


    Constructors :
        `from_yolo(yolo_line: str, image_width: int, image_height: int, id_to_category: Dict[int, str])`:
            Constructs a Keypoint from a line in a YOLO formatted labels file. It requires the original image dimensions and a dictionary mapping category IDs to category names.
            **Note:** This method ignores the "visibility" information (denoted by 'v') in the YOLO format.


    Properties :
        `category` (str):
            The category of the object the keypoint belongs to (inherited from the `bounding_box`).
        `center` (Tuple[float]):
            The (x, y) coordinates of the bounding box's center (inherited from the `bounding_box`).
        `box` (Tuple[float]):
            A list containing the bounding box coordinates as [left, top, right, bottom] (inherited from the `bounding_box`).


    Methods :
        `to_yolo(self, image_width: int, image_height: int, category_to_id: Dict[str, int]) -> str`:
            Generates a YOLO formatted string representation of this `Keypoint` object. It requires the image dimensions and a dictionary mapping category strings to integer labels.
        `validate_keypoint(cls, bounding_box: BoundingBox, keypoint: Point) -> None`:
            Validates that a keypoint lies within the specified bounding box. Raises a ValueError if the keypoint is outside the bounding box.
    """

    keypoint: Point
    bounding_box: BoundingBox

    def __init__(self, keypoint: Point, bounding_box: BoundingBox):
        """Overrides the default constructor from dataclass to validate the parameters before constructing."""
        Keypoint.validate_keypoint(bounding_box, keypoint)
        self.keypoint = keypoint
        self.bounding_box = bounding_box

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
            `yolo_line` (str):
                A string in the yolo label format (c x y w h kpx kpy v).
            `image_width` (int):
                The original image's width.
            `image_height` (int):
                The original image's height.
            `id_to_category` (Dict):
                A dictionary that maps the id number in the label to the category.

        Returns:
            A `BoundingBox` object containing the yolo_line's data.
        """
        bounding_box = BoundingBox.from_yolo(
            yolo_line, image_width, image_height, id_to_category
        )
        keypoint_x = float(yolo_line.split()[5])
        keypoint_y = float(yolo_line.split()[6])
        keypoint = Point(keypoint_x * image_width, keypoint_y * image_height)
        return Keypoint(keypoint, bounding_box)

    @classmethod
    def validate_keypoint(cls, bounding_box: BoundingBox, keypoint: Point) -> None:
        """Validates that a keypoint lies within the specified bounding box.

        This classmethod ensures that the `keypoint` (represented by a `Point` object)
        falls within the confines of the provided `bounding_box` (represented by a
        `BoundingBox` object). It checks both the x and y coordinates of the keypoint
        against the left, top, right, and bottom boundaries of the bounding box.

        Args:
            bounding_box:
                The `BoundingBox` object representing the enclosing region.
            keypoint:
                The `Point` object representing the keypoint to be validated.

        Raises:
            ValueError: If the keypoint's coordinates are not within the bounding box.
        """
        in_bounds_x: bool = bounding_box.left <= keypoint.x <= bounding_box.right
        in_bounds_y: bool = bounding_box.top <= keypoint.y <= bounding_box.bottom
        in_bounds: bool = in_bounds_x and in_bounds_y
        if not in_bounds:
            raise ValueError(
                f"Keypoint is not in the bounding box intended to enclose it (Keypoint:{(keypoint.x, keypoint.y)}, BoundingBox:{str(bounding_box)})"
            )

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

    def set_box(
        self, new_left: int, new_top: int, new_right: int, new_bottom: int
    ) -> BoundingBox:
        """Sets this Keypoints's BoundingBox's values for left, top, right, bottom.

        Args:
            new_left (int):
                The new left side for the box.
            new_top (int):
                The new top side for the box.
            new_right (int):
                The new right side for the box.
            new_bottom (int):
                The new bottom side for the box.

        Returns: A new Keypoint with a new bounding box.
        """
        return Keypoint(
            point=self.point,
            bounding_box=self.bounding_box.set_box(
                new_left, new_top, new_right, new_bottom
            ),
        )

    def set_keypoint(self, new_x: int, new_y: int) -> "Keypoint":
        """Sets this Keypoint's Keypoint to a new point.

        Args:
            new_x (int):
                The new x value for the Keypoint.
            new_y (int):
                The new y value for the Keypoint.

        Returns: A new Keypoint with a new Point as its keypoint.
        """
        self.validate_keypoint(self.bounding_box, Point(new_x, new_y))
        return Keypoint(Point(new_x, new_y), self.bounding_box)

    def to_yolo(
        self, image_width: int, image_height: int, category_to_id: Dict[str, int]
    ) -> str:
        """Writes the data from this `Keypoint` into a yolo formatted string.

        Args :
            `image_width` (int):
                The image's width that this `Keypoint` belongs to.
            `image_height` (int):
                The image's height that this `Keypoint` belongs to.
            `category_to_id` (Dict[str, int]):
                A dictionary that maps the category string to an id (int).

        Returns:
            A string that encodes this `Keypoint`'s data for a single line in a yolo label file.
        """
        yolo_line = self.bounding_box.to_yolo(image_width, image_height, category_to_id)
        keypoint_x, keypoint_y = (
            self.keypoint.x / image_width,
            self.keypoint.y / image_height,
        )
        yolo_line += f" {keypoint_x} {keypoint_y}"
        return yolo_line
