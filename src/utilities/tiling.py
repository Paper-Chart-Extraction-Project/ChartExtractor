"""This module provides functions for splitting an image and its annotations into smaller, overlapping tiles.

The `tile_image` function splits a PIL `Image` object into a grid of tiles with a specified size and overlap ratio. 
It handles padding the image with black pixels if necessary to ensure all tiles fit perfectly within the image boundaries.

The `tile_annotations` function is a sister function to `tile_image` and can be used to tile annotations associated with the 
image using (nearly) the same parameters as `tile_image`. It assumes annotations implement a `'box'` property representing 
their location within the image. Each annotation is assigned to the tile(s) that completely enclose its bounding box.
"""

from PIL import Image
from typing import List, Tuple, Union
import math
from utilities.annotations import BoundingBox, Keypoint


def tile_image(
    image: Image.Image,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
) -> List[Image.Image]:
    """Splits a larger image into smaller 'tiles'.

    In the likely event that the exact choices of overlap ratios and slice dimensions do not
    multiply to make exactly the image's dimensions, the image.crop method pads the image with
    black on the right and bottom sides.

    Args:
        `image` (PIL Image):
            The image to tile.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Returns:
        A list of sliced images.
    """
    validate_tile_parameters(
        image,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    image_width, image_height = image.size
    tile_coordinates: List[Tuple[int, int, int, int]] = generate_tile_coordinates(
        image_width,
        image_height,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    images: List[Image.Image] = [
        [image.crop(box) for box in tc] for tc in tile_coordinates
    ]
    return images


def validate_tile_parameters(
    image: Image.Image,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
) -> None:
    """Validates the parameters for the function 'tile_image'.

    Args:
        `image` (PIL Image):
            The image to tile.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Raises:
        ValueError:
            If slice_width is not within (0, image_width], slice_height not within (0, image_height), or horizontal/vertical overlap ratio not in (0, 1].
    """
    if not 0 < slice_width <= image.size[0]:
        raise ValueError(
            f"slice_width must be between 1 and the image's width (slice_width passed was {slice_width})."
        )
    if not 0 < slice_height <= image.size[1]:
        raise ValueError(
            f"slice_height must be between 1 and the image's height (slice_height passed was {slice_height})."
        )
    if not 0 < horizontal_overlap_ratio <= 1:
        raise ValueError(
            f"horizontal_overlap_ratio must be greater than 0 and less than or equal to 1 (horizontal_overlap_ratio passed was {horizontal_overlap_ratio}."
        )
    if not 0 < vertical_overlap_ratio <= 1:
        raise ValueError(
            f"vertical_overlap_ratio must be greater than 0 and less than or equal to 1 (vertical_overlap_ratio passed was {vertical_overlap_ratio}."
        )


def generate_tile_coordinates(
    image_width: int,
    image_height: int,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: int,
    vertical_overlap_ratio: int,
) -> List[List[Tuple[int, int, int, int]]]:
    """Generates the box coordinates of the tiles for the function 'tile_image'.

    Args:
        `image_width` (int):
            The image's width.
        `image_height` (int):
            The image's height.
        `slice_height` (int):
            The height of each slice.
        `slice_width` (int):
            The width of each slice.
        `horizontal_overlap_ratio` (float):
            The amount of left-right overlap between slices.
        `vertical_overlap_ratio` (float):
            The amount of top-bottom overlap between slices.

    Returns:
        A 2d list of four coordinate tuples encoding the left, top, right, and bottom of each tile.
    """
    tile_coords: List[List[Tuple[int, int, int, int]]] = [
        [
            (
                x * round(slice_width * horizontal_overlap_ratio),
                y * round(slice_width * vertical_overlap_ratio),
                slice_width + x * round(slice_width * horizontal_overlap_ratio),
                slice_height + y * round(slice_width * vertical_overlap_ratio),
            )
            for x in range(
                math.floor(image_width / (slice_width * horizontal_overlap_ratio))
            )
        ]
        for y in range(
            math.floor(image_height / (slice_height * vertical_overlap_ratio))
        )
    ]
    return tile_coords


def tile_annotations(
    annotations: List[Union[BoundingBox, Keypoint]],
    image_width: int,
    image_height: int,
    slice_width: int,
    slice_height: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
):
    """Tiles image annotations based on a specified grid pattern with overlap.

    This function takes a list of annotations (any annotation that implements the 'box' property)
    representing objects within an image, and divides the image into a grid of tiles
    with a specified size and overlap. It then assigns each annotation to the tile(s)
    based on whether the annotation fully fits into the tile.

    Args:
        `annotations` (List[Union[BoundingBox, Keypoint]]):
            A list of annotations (anything that implements the 'box' property).
        `image_width` (int):
            The width of the image in pixels.
        `image_height` (int):
            The height of the image in pixels.
        `slice_width` (int):
            The width of each tile in pixels.
        `slice_height` (int):
            The height of each tile in pixels.
        `horizontal_overlap_ratio` (float):
            The ratio (0.0 to 1.0) of the tile width that overlaps horizontally between adjacent tiles.
        `vertical_overlap_ratio` (float):
            The ratio (0.0 to 1.0) of the tile height that overlaps vertically between adjacent tiles.

    Returns:
        A list of lists, where each sub-list represents a tile in the grid. Each tile's
        sub-list contains the annotations that intersect fully with that specific tile.
    """
    tile_coordinates: List[List[Tuple[int, int, int, int]]] = generate_tile_coordinates(
        image_width,
        image_height,
        slice_width,
        slice_height,
        horizontal_overlap_ratio,
        vertical_overlap_ratio,
    )
    annotation_tiles = [
        [get_annotations_in_tile(annotations, tc) for tc in tc_list]
        for tc_list in tile_coordinates
    ]
    annotation_tiles = [
        [
            [
                (
                    correct_bounding_box(
                        ann, tile_coordinates[iy][ix][0], tile_coordinates[iy][ix][1]
                    )
                    if isinstance(ann, BoundingBox)
                    else correct_keypoint(
                        ann, tile_coordinates[iy][ix][0], tile_coordinates[iy][ix][1]
                    )
                )
                for ann in tile_anns
            ]
            for ix, tile_anns in enumerate(row_anns)
        ]
        for iy, row_anns in enumerate(annotation_tiles)
    ]
    return annotation_tiles


def get_annotations_in_tile(
    annotations: List[Union[BoundingBox, Keypoint]], tile: Tuple[int, int, int, int]
) -> List:
    """Filters annotations that fully intersect with a given tile.

    This function takes a list of annotations (assumed to implement the 'box' property)
    and a tile represented by its top-left and bottom-right corner coordinates `(left, top, right, bottom)`
    as a tuple. It returns a new list containing only the annotations that have a bounding box
    intersecting with the specified tile area.

    Args:
        `annotations`:
            A list of annotations (expected to implement the 'box' property).
        `tile` (Tuple[int, int, int, int]):
            A tuple representing the tile's bounding box coordinates `(left, top, right, bottom)`.

    Returns:
        A list of `BoundingBox` objects that intersect with the specified tile.
    """
    annotation_in_tile = lambda ann, tile: all(
        [
            ann.box[0] >= tile[0],
            ann.box[1] >= tile[1],
            ann.box[2] <= tile[2],
            ann.box[3] <= tile[3],
        ]
    )
    annotations_in_tile: List = list(
        filter(lambda ann: annotation_in_tile(ann, tile), annotations)
    )
    return annotations_in_tile


def correct_bounding_box(
    annotation: Union[BoundingBox, Keypoint],
    tile_left: int,
    tile_top: int,
) -> BoundingBox:
    """Corrects the location of a bounding box given the tile's left and top relative to the entire image.

    Args:
        `annotation` (Union[BoundingBox, Keypoint]):
            The annotation to correct.
        `tile_left` (int):
            The tile's left side coordinate relative to the entire untiled image.
        `tile_top` (int):
            The tile's left side coordinate relative to the entire untiled image.

    Returns: A BoundingBox with new coordinates that fit within the new tile.
    """
    return annotation.set_box(
        annotation.box[0] - tile_left,
        annotation.box[1] - tile_top,
        annotation.box[2] - tile_left,
        annotation.box[3] - tile_top,
    )


def correct_keypoint(annotation: Keypoint, tile_left: int, tile_top: int) -> Keypoint:
    """Corrects the location of a keypoint given the tile's left and top relative to the entire image.

    Args:
        `annotation` (Keypoint):
            The Keypoint annotation to correct.
        `tile_left` (int):
            The tile's left side coordinate relative to the entire untiled image.
        `tile_top` (int):
            The tile's left side coordinate relative to the entire untiled image.

    Returns: A Keypoint with new coordinates that fit within the new tile.
    """
    annotation = correct_bounding_box(annotation, tile_left, tile_top)
    return annotation.set_keypoint(
        annotation.point.x - tile_left, annotation.point.y - tile_top
    )
