"""Module for remapping points using a homography transform.

This module exposes a two functions, (1) find_homography, which is a thin wrapper around opencv's
findHomography function so that cv2 doesn't have to be imported where this is used, and
(2) transform_point, which takes a point and a homography matrix and transforms the point.

Functions:
    find_homography(source_points: List[Tuple[int, int]], destination_points: List[Tuple[int, int]])
        -> np.ndarray:
        Computes the homography transformation that maps the source_points array to the
        destination_points array. A thin wrapper around opencv's findHomography function.
    transform_point(point: Tuple[int, int], homography_matrix: np.ndarray) -> Tuple[int, int]:
        Remaps a single point using the homography matrix.
"""

import cv2
import numpy as np
from typing import List, Tuple


def find_homography(
    source_points: List[Tuple[int, int]],
    destination_points: List[Tuple[int, int]],
) -> np.ndarray:
    """A thin wrapper around opencv's findHomography function.
    
    Args:
        source_points (List[Tuple[int, int]]):
            The points to move to match to destination points.
        destination_points (List[Tuple[int, int]]):
            The points that the source points are moved to match.

    Returns:
        A numpy ndarray containing the homography matrix which can be used with transform_point
        to transform points according to the transformation that remaps the source points to the
        destination points.
    """
    pass


def transform_point(
    point: Tuple[int, int],
    homography_matrix: np.ndarray,
) -> Tuple[int, int]:
    """Remaps a single point using the homography matrix.
    
    Args:
        point (Tuple[int, int]):
            The point to remap.
        homography_matrix (np.ndarray):
            A homography matrix.

    Returns:
        A point which has been transformed by the homography.
    """
    pass
