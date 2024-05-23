"""The tiling.py module implements a function called 'tile_image' which slices images into smaller 
pieces called 'tiles'. This is used to improve the accuracy and memory efficiency of object
detection models that need to detect very small objects in high resolution images.
"""


from PIL import Image
from typing import List


def tile_image(
    image: Image.Image,
    slice_height: int,
    slice_width: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
) -> List[Image.Image]:
    """Splits a larger image into smaller 'tiles'.

    In the likely event that the exact choices of overlap ratios and slice dimensions do not
    multiply to make exactly the image's dimensions, the rightmost and bottommost column/row
    of tiles will simply slide until it hits the right/bottom of the image regardless of whether
    or not it equals the overlap ratio.

    Args : 
        image (PIL Image) - The image to tile.
        slice_height (int) - The height of each slice.
        slice_width (int) - The width of each slice.
        horizontal_overlap_ratio (float) - The amount of left-right overlap between slices.
        vertical_overlap_ratio (float) - The amount of top-bottom overlap between slices.

    Returns : A list of sliced images.
    """
    validate_tile_parameters(image, slice_height, slice_width, horizontal_overlap_ratio, vertical_overlap_ratio)


def validate_tile_parameters(
    image: Image.Image,
    slice_height: int,
    slice_width: int,
    horizontal_overlap_ratio: float,
    vertical_overlap_ratio: float,
):
    """Validates the parameters for the function 'tile_image'.

    Args : 
        image (PILImage) - The image to tile.
        slice_height (int) - The height of each slice.
        slice_width (int) - The width of each slice.
        horizontal_overlap_ratio (float) - The amount of left-right overlap between slices.
        vertical_overlap_ratio (float) - The amount of top-bottom overlap between slices.
    """
    if not 0 < slice_width <= image.size[0]:
        raise ValueError(f"slice_width must be between 1 and the image's width \
                           (slice_width passed was {slice_width}).")
    if not 0 < slice_height <= image.size[1]:
        raise ValueError(f"slice_height must be between 1 and the image's height \
                           (slice_height passed was {slice_height}).")
    if not 0 < horizontal_overlap_ratio <= 1:
        raise ValueError(f"horizontal_overlap_ratio must be greater than 0 and less than or equal to 1 \
                           (horizontal_overlap_ratio passed was {horizontal_overlap_ratio}.")
    if not 0 < vertical_overlap_ratio <= 1:
        raise ValueError(f"vertical_overlap_ratio must be greater than 0 and less than or equal to 1 \
                           (vertical_overlap_ratio passed was {vertical_overlap_ratio}.")
                           