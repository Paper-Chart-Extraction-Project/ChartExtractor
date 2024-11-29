"""
Module that can be called to perform TPS on images. This should be called after homography.

TPS is checked via 2 methods:

1. The first method is to compare the points we want to use for TPS with their expected locations.
    If further than a reasonably small threshold, given we have already done homography, then we do not use this point.
2. The second method is to use RANSAC to further filter out points that may be incorrect.

We may later incorporate a function that utilizes the confidence of our detection models of certain objects to further filter out points.
"""

# Built-in imports
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

# Internal imports
from utilities.annotations import BoundingBox

# External imports
import cv2
import numpy as np
from collections import Counter
from scipy.interpolate import Rbf

# Function to create a list of BoundingBox objects from a landmarks
# load introp_document_landmarks.json which will be used as dst_points
PATH_TO_LANDMARKS = os.path.join(
    os.path.dirname(__file__), "data", "intraop_document_landmarks.json"
)

DESIRED_IMAGE_WIDTH = 800
DESIRED_IMAGE_HEIGHT = 600


def __label_studio_to_bboxes(path_to_json_data: Path) -> List[BoundingBox]:
    """
    Convert the json data from label studio to a list of BoundingBox objects

    Args:
        path_to_json_data (Path): Path to the json data from label studio

    Returns:
        List[BoundingBox]: List of BoundingBox objects
    """
    json_data: List[Dict] = json.loads(open(str(path_to_json_data)).read())
    return {
        sheet_data["data"]["image"].split("-")[-1]: [
            BoundingBox(
                category=label["value"]["rectanglelabels"][0],
                left=label["value"]["x"] / 100 * DESIRED_IMAGE_WIDTH,
                top=label["value"]["y"] / 100 * DESIRED_IMAGE_HEIGHT,
                right=(label["value"]["x"] / 100 + label["value"]["width"] / 100)
                * DESIRED_IMAGE_WIDTH,
                bottom=(label["value"]["y"] / 100 + label["value"]["height"] / 100)
                * DESIRED_IMAGE_HEIGHT,
            )
            for label in sheet_data["annotations"][0]["result"]
        ]
        for sheet_data in json_data
    }


# Start with turning Hannah's code into a function
def __filter_by_distance(
    src_bbs: List[BoundingBox],
    dst_bbs: List[BoundingBox],
    threshold: float,
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """
    Filter out source and destination points that have a distance greater than the threshold.
    Large transformations are likely to be outliers and erroneous. We only expect small tweaks via the thin plate spline transformation.
    Homography should already be completed prior to TPS.

    Args:
        src_bbs (List[BoundingBox]): The source points
        dst_bbs (List[BoundingBox]): The destination points
        threshold (float): The threshold distance

    Returns:
        Tuple[List[BoundingBox], List[BoundingBox]]: The filtered source and destination points

    """
    filtered_points = [
        (src_bb, dst_bb)
        for src_bb, dst_bb in zip(src_bbs, dst_bbs)
        if abs(src_bb.top - dst_bb.top) < threshold
        and abs(src_bb.left - dst_bb.left) < threshold
    ]

    new_src_bbs, new_dst_bbs = zip(*filtered_points) if filtered_points else ([], [])

    return list(new_src_bbs), list(new_dst_bbs)


# Now we can turn Matt's RANSAC code into a function
def __filter_by_RANSAC(
    src_bbs: List[BoundingBox],
    dst_bbs: List[BoundingBox],
    threshold: float,
    max_iters: int = 5000,
    confidence_limit: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out source and destination points that are not inliers according to RANSAC

    Args:
        src_bbs (np.ndarray): The source points
        dst_bbs (np.ndarray): The destination points
        threshold (float): The threshold distance
        max_iters (int, optional): The maximum number of iterations for RANSAC. Defaults to 5000.
        confidence_limit (float, optional): The confidence limit for RANSAC. Defaults to 0.99.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The filtered source and destination points as numpy arrays in this order (src_x, src_y, dst_x, dst_y)
    """
    # Turn the BoundingBox objects into numpy arrays of coordinates
    src_points = np.array([[bb.left, bb.top] for bb in src_bbs], dtype=np.float32)
    dst_points = np.array([[bb.left, bb.top] for bb in dst_bbs], dtype=np.float32)

    # Complete RANSAC
    _, mask = cv2.findHomography(
        dst_points,
        src_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=max_iters,
        confidence=confidence_limit,
    )
    inlier_mask = mask.ravel() == 1

    # Apply the mask to the source and destination points
    filtered_src = src_points[inlier_mask]
    filtered_dst = dst_points[inlier_mask]

    # Get the x and y coordinates of the filtered points
    src_x, src_y = filtered_src[:, 0], filtered_src[:, 1]
    dst_x, dst_y = filtered_dst[:, 0], filtered_dst[:, 1]

    return src_x, src_y, dst_x, dst_y


# Now we need to merge the two functions into one that uses RANSAC to filter out outliers as well as those far from the destination points
def transform_thin_plate_splines(
    image: np.ndarray,
    src_bbs: List[BoundingBox],
    max_dist: float = 4.0,
    threshold: float = 10.0,
    max_iters: int = 5000,
    confidence_limit: float = 0.99,
) -> np.ndarray:
    """
    Perform a thin plate spline transformation on the image using the src_bbs and dst_bbs, using RANSAC to filter out outliers.
    We assume that homography was completed prior to calling this function.
    We start by filtering by points that are too far from their destination counterparts, then we use RANSAC to filter out outliers.

    Args:
        image (np.ndarray): The image to be transformed
        src_bbs (List[BoundingBox]): List of BoundingBox objects
        max_dist (float, optional): The maximum distance for filtering out points. Defaults to 4.0.
        threshold (float, optional): The threshold distance for RANSAC. Defaults to 10.0.
        max_iters (int, optional): The maximum number of iterations for RANSAC. Defaults to 5000.
        confidence_limit (float, optional): The confidence limit for RANSAC. Defaults to 0.99.

    Returns:
        np.ndarray: The transformed image
    """
    # Get landmarks from the json file
    landmark_location_data: Dict[str, List[BoundingBox]] = __label_studio_to_bboxes(
        PATH_TO_LANDMARKS
    )
    # Extract relevant ones
    dst_bbs = landmark_location_data[
        "unified_intraoperative_preoperative_flowsheet_v1_1_front.png"
    ]

    del landmark_location_data

    # Get the categories from dst_bbs
    landmark_cats = [bb.category for bb in dst_bbs]
    # remove all bbs in src that are not in those categories
    src_bbs = [bb for bb in src_bbs if bb.category in landmark_cats]
    # get list of duplicate keys
    duplicate_count_src = dict(Counter([bb.category for bb in src_bbs]))
    duplicates = [k for k, v in duplicate_count_src.items() if v > 1]
    duplicate_count_dst = dict(Counter([bb.category for bb in dst_bbs]))
    duplicates.extend([k for k, v in duplicate_count_dst.items() if v > 1])
    duplicates = list(set(duplicates))
    # print(duplicates)
    # remove duplicates
    src_bbs = [bb for bb in src_bbs if bb.category not in duplicates]
    dst_bbs = [bb for bb in dst_bbs if bb.category not in duplicates]

    # sort categories alphabetically
    src_bbs = sorted(src_bbs, key=lambda bb: bb.category)
    dst_bbs = sorted(dst_bbs, key=lambda bb: bb.category)

    # remove source points with suspiciously high distances to their destination counterparts
    src_bbs, dst_bbs = __filter_by_distance(src_bbs, dst_bbs, max_dist)

    # Now lets use RAANSAC to filter out outliers
    src_x, src_y, dst_x, dst_y = __filter_by_RANSAC(
        src_bbs, dst_bbs, threshold, max_iters, confidence_limit
    )

    # use RBF function to do the thin plate splines
    rbf_x = Rbf(dst_x, dst_y, src_x, function="thin_plate")
    rbf_y = Rbf(dst_x, dst_y, src_y, function="thin_plate")

    # Alter the image according to the transformation
    h, w, _ = image.shape
    # create grid
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    grid_x, grid_y = np.meshgrid(x, y)

    # apply the transformation
    # reshape into grid
    transformed_x = rbf_x(grid_x, grid_y).astype(np.float32)
    transformed_y = rbf_y(grid_x, grid_y).astype(np.float32)

    transformed_x = np.clip(transformed_x, 0, image.shape[1] - 1)
    transformed_y = np.clip(transformed_y, 0, image.shape[0] - 1)

    # warp the image
    warp_img = cv2.remap(
        image, transformed_x, transformed_y, interpolation=cv2.INTER_LINEAR
    )

    return warp_img
