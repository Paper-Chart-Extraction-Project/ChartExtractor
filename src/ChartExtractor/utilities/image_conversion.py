"""Converts between PIL and OpenCV image formats.

This module provides functions to convert between Python Imaging Library (PIL)
image format and OpenCV image format.
"""

import cv2
from PIL import Image
import numpy as np


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts a PIL image to OpenCV image format.

    Args:
        pil_image (Image.Image):
            A PIL image object.

    Returns:
        A NumPy array representing the image in OpenCV format (BGR channel order).

    Raises:
        ValueError:
            If the input image mode is not compatible with RGB or BGR.
    """
    cv2_image = np.array(pil_image)
    if pil_image.mode not in ("RGB", "BGR"):
        raise ValueError(
            f"Unsupported image mode: {pil_image.mode}. Only RGB and BGR modes are supported."
        )
    if pil_image.mode == "RGB":
        cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts an OpenCV image to PIL image format.

    Args:
        cv2_image (np.ndarray):
            A NumPy array representing an image in OpenCV format (BGR channel order).

    Returns:
        A PIL image object.
    """
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img)
    return pil_image
