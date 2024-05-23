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
    pass


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
    pass
