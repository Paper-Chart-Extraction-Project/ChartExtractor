"""Module for remapping points using a homography transform.

This module exposes two functions, (1) find_homography, which is a thin wrapper around opencv's
findHomography function that restricts the original function's usage to only 2d points, and
provides more robust error messages for this libraries usage, and (2) transform_point, which takes 
a point and a homography matrix and transforms the point.

Functions:
    find_homography(source_points: List[Tuple[int, int]], destination_points: List[Tuple[int, int]])
        -> np.ndarray:
        Computes the homography transformation that maps the source_points array to the
        destination_points array. A thin wrapper around opencv's findHomography function.
    transform_point(point: Tuple[int, int], homography_matrix: np.ndarray) -> Tuple[int, int]:
        Remaps a single point using the homography matrix.
"""

# Built-in imports
from typing import List, Tuple

# Internal imports
from ..utilities.annotations import BoundingBox

# External imports
import cv2
import numpy as np


def find_homography(
    source_points: List[Tuple[int, int]],
    destination_points: List[Tuple[int, int]],
) -> np.ndarray:
    """A thin wrapper around opencv's findHomography function.

    Provides some additional checks and more informative errors.
    
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
    too_few_source_points: bool = len(source_points) < 4
    too_few_destination_points: bool = len(destination_points) < 4
    unequal_point_sets: bool = len(source_points) != len(destination_points)
    source_points_not_two_dimensional: bool = set([len(p) for p in source_points]) == {2}
    destination_points_not_two_dimensional: bool = set([len(p) for p in destination_points]) == {2}

    if too_few_source_points:
        raise ValueError(
            f"Too few points in source set (need at least 4, had {len(source_points)})."
        )
    if too_few_destination_points:
        raise ValueError(
            f"Too few points in destination set (need at least 4, had {len(destination_points)})."
        )
    if unequal_point_sets:
        err_msg: str = "Point sets were unequal in length. "
        err_msg += f"(length of source: {len(source_points)}, "
        err_msg += f"length of destination: {len(destination_points)})"
        raise ValueError(err_msg)
    if source_points_not_two_dimensional:
        err_msg: str = "Source point set contains non two dimensional points. "
        err_msg += f"(Included dimensions: {set([len(p) for p in source_points])})"
        raise ValueError(err_msg)
    if destination_points_not_two_dimensional:
        err_msg: str = "Destination point set contains non two dimensional points. "
        err_msg += f"(Included dimensions: {set([len(p) for p in destination_points])})"
        raise ValueError(err_msg)
    
    return findHomography(source_points, destination_points)


def transform_point(point: Tuple[int, int], homography_matrix: np.ndarray) -> Tuple[int, int]:
    """Remaps a single point using the homography matrix.
    
    Args:
        point (Tuple[int, int]):
            The point to remap.
        homography_matrix (np.ndarray):
            A homography matrix.

    Returns:
        A point which has been transformed by the homography.
    """
    if len(point) != 2:
        raise ValueError(f"Point is not two dimensional: {point}.")
    
    remapped_point = homography_matrix.dot(np.array([point[0], point[1], 1]))
    remapped_point /= remapped_point[2]
    return (remapped_point[0], remapped_point[1])


def transform_box(box: BoundingBox, homography_matrix: np.ndarray) -> BoundingBox:
    """Remaps a BoundingBox using the homography matrix.

    Args:
        box (BoundingBox):
            The bounding box to remap.
        homography_matrix (np.ndarray):
            A homography matrix
    """
    remapped_top_left: Tuple[float, float] = transform_point((box.left, box.top), homography_matrix)
    remapped_top_right: Tuple[float, float] = transform_point(
        (box.right, box.top),
        homography_matrix,
    )
    remapped_bottom_left: Tuple[float, float] = transform_point(
        (box.left, box.bottom),
        homography_matrix,
    )
    remapped_bottom_right: Tuple[float, float] = transform_point(
        (box.right, box.bottom),
        homography_matrix,
    )
    
    left = min(remapped_top_left[0], remapped_bottom_left[0])
    top = min(remapped_top_left[1], remapped_top_right[1])
    right = max(remapped_top_right[0], remapped_bottom_right[0])
    bottom = max(remapped_bottom_left[1], remapped_bottom_right[1])

    return BoundingBox(box.category, left, top, right, bottom)
    
