"""Performs homography transformation on images.

This module provides a function for performing homography transformation on
images. Homography is a technique used to map points from one image (source)
to another image (destination) based on corresponding point locations.
The homography is a linear mapping, so non-linear distortions in the image
cannot be resolved by the homography alone.

The `homography_transform` function takes a source PIL image, corresponding
source and destination point lists, and an original image size as input.
It then:

    1. Converts source and destination points to NumPy arrays.
    2. Validates that both point lists have the same number of elements
       (and at least 4 points for homography calculation).
    3. Converts the source PIL image to OpenCV format.
    4. Calculates the homography matrix using `cv2.findHomography`.
    5. Warps the source image to the destination perspective using
       `cv2.warpPerspective`.
    6. Converts the resulting OpenCV image back to PIL format.
"""

from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from utilities.image_conversion import pil_to_cv2, cv2_to_pil


def homography_transform(
    src_image: Image.Image,
    src_points: List[Tuple[float, float]],
    dest_points: List[Tuple[float, float]],
    original_image_size: Tuple[float, float] = (3300, 2250),
) -> Image.Image:
    """Performs homography transformation on an image.

    This function transforms an image (src_image) based on corresponding points
    between the source and destination images. It calculates the homography matrix
    and uses it to warp the source image to the perspective of the destination points.

    Args:
        src_image (Image.Image):
            A PIL image object representing the source image.
        src_points (List[Tuple[int, int]]):
            A list of tuples (x, y) representing points in the source image that correspond
            to points in the destination image.
        dest_points (List[Tuple[int, int]]):
            A list of tuples (x, y) representing points in the destination image that points
            in the source image correspond to (where the source image should be warped to).
        original_image_size (Tuple[float, float]):
            A tuple (width, height) representing the size of the control image.
            Defaults to (3300, 2250).

    Returns:
        A PIL image object representing the transformed source image.

    Raises:
        ValueError:
            If the length of src_points and dest_points don't match (must have the same
            number of corresponding points), or if there are less than 4 points.
    """
    src_points: np.ndarray = np.array(src_points)
    dest_points: np.ndarray = np.array(dest_points)

    if len(src_points) != len(dest_points):
        raise ValueError(
            "Source and destination points must have the same number of elements."
        )
    if len(src_points) < 4 or len(dest_points) < 4:
        raise ValueError("Must have 4 or more points to compute the homography.")

    src_image = pil_to_cv2(src_image)
    h, _ = cv2.findHomography(src_points, dest_points)
    dest_image = cv2.warpPerspective(src_image, h, original_image_size)
    dest_image = cv2_to_pil(dest_image).resize(330, 225)
    return dest_image
