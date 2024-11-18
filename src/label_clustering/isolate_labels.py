"""Module that contains code for isolating labels from a list of bounding boxes from the blood pressure section.

This module contains functions for isolating labels from a list of bounding boxes.
It includes a public function for external use and several worker functions that do the isolation.
"""

# Built-in imports
from pathlib import Path
from typing import List, Tuple

# External imports
import cv2
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

# Internal imports
from ..utilities.annotations import BoundingBox


def __find_density_max(values: List[int], search_area: int) -> int:
    """
    Given a list of values and a search area, find the index of where the highest density is.
    The list of values correspond to identifying points for the bounding boxes and the search area corresponds to the images height or width.

    Args:
        values: List of identifying points for the bounding boxes
        search_area: height/width of the image dependent on whether x or y axis is being search.

    Returns:
        The axis value that has the highest density of bounding boxes.
    """
    kde = gaussian_kde(values, bw_method=0.2)

    x_values = np.linspace(0, search_area, 10000)

    kde_vals = kde(x_values)

    max_index = np.argmax(kde_vals)
    return x_values[max_index]


def __remove_bb_outliers(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Given a list of bounding boxes, remove the outliers from the x axis, then remove the outliers from the y axis

    Args:
        boxes: List of Bounding Boxes to filter

    Returns:
        Filtered list of Bounding Boxes
    """
    x_vals = [bb.left for bb in boxes]
    # find the 25th percentile
    x_Q1 = np.percentile(x_vals, 25)
    # find the 75th percentile
    x_Q3 = np.percentile(x_vals, 75)
    # find the IQR
    x_IQR = x_Q3 - x_Q1
    # determine lower and upper bounds
    x_lower = x_Q1 - 1.5 * x_IQR
    x_upper = x_Q3 + 1.5 * x_IQR
    # remove outliers via the x axis
    x_filtered = [bb for bb in boxes if x_lower <= bb.left <= x_upper]

    y_vals = [bb.top for bb in x_filtered]
    # find the 25th percentile
    y_Q1 = np.percentile(y_vals, 25)
    # find the 75th percentile
    y_Q3 = np.percentile(y_vals, 75)
    # find the IQR
    y_IQR = y_Q3 - y_Q1
    # determine the lower and upper bounds
    y_lower = y_Q1 - 1.5 * y_IQR
    y_upper = y_Q3 + 1.5 * y_IQR
    # remove outliers via the y axis
    filtered = [bb for bb in x_filtered if y_lower <= bb.top <= y_upper]

    return filtered


def extract_relevant_bounding_boxes(
    sheet_data: List[str],
    path_to_image: Path,
    show_images: bool = False,
    desired_img_width: int = 800,
    desired_img_height: int = 600,
) -> Tuple[List[str], List[str]]:
    """
    Given sheet data for bounding boxes in YOLO format, find the bounding boxes corresponding to the number and time on the BP chart.
    Return the bounding boxes that are within the selected region split into two lists: time labels and numerical values.

    Args:
        sheet_data: List of bounding boxes in YOLO format.
        path_to_image: Path to the image file.
        show_images: Boolean flag to show the image with the selected region and bounding boxes. Default is False.
        desired_img_width: Desired width of the image to display. Default is 800.
        desired_img_height: Desired height of the image to display. Default is 600.


    Returns:
        Tuple of Lists of string representations of bounding boxes that are within the selected region, in YOLO format.
        The first list contains bounding boxes in the top-right region -- representing time labels.
        The second list contains bounding boxes in the bottom-left region -- representing numerical values for mmHg and bpm.
            (bounding_boxes_time, bounding_boxes_numbers)
    """

    # Load the image
    image = cv2.imread(path_to_image)

    # Display the image and allow the user to select a ROI
    resized_image = cv2.resize(image, (desired_img_width, desired_img_height))

    # convert the YOLO data to Bounding Boxes
    bboxes: List[BoundingBox] = [
        BoundingBox.from_yolo(yolo_bb, desired_img_width, desired_img_height)
        for yolo_bb in sheet_data
    ]

    # generate a list of the digit categories
    digit_categories: List[str] = [str(i) for i in range(10)]

    # filter out non bounding boxes and those whose category is not a digit
    bboxes: List[BoundingBox] = list(
        filter(
            lambda bb: isinstance(bb, BoundingBox) and bb.category in digit_categories,
            bboxes,
        )
    )

    # find the point with the maximum density of bounding boxes
    bboxes_right: List[int] = [bb.right for bb in bboxes]
    # x_loc is the vertical line to the left of the time axis and right of the numbers axis
    x_loc: int = __find_density_max(bboxes_right, desired_img_width)

    bboxes_bottom: List[int] = [bb.bottom for bb in bboxes]
    # y_loc is the horizontal line undert the time axis and above the number axis
    y_loc: int = __find_density_max(bboxes_bottom, desired_img_height)

    bounding_boxes_time = []
    bounding_boxes_numbers = []

    # Process the bounding boxes
    for bounding_box in bboxes:
        # get the center point of the bounding box for comparison
        x_center_bb, y_center_bb = bounding_box.center

        # check if the bounding box is a number on the BP chart by comparing to the KDE index + a threshold
        if x_center_bb > x_loc - 15 and x_center_bb < x_loc + 2:
            bounding_boxes_numbers.append(bounding_box)
        # check if the bounding box is a time on the BP chart by comparing to the KDE index + a threshold
        elif y_center_bb > y_loc - 10 and y_center_bb < y_loc + 2:
            bounding_boxes_time.append(bounding_box)

    bounding_boxes_numbers = __remove_bb_outliers(bounding_boxes_numbers)
    bounding_boxes_time = __remove_bb_outliers(bounding_boxes_time)

    for bounding_box in bounding_boxes_numbers:
        x_min = int(bounding_box.left)
        x_max = int(bounding_box.right)
        y_min = int(bounding_box.top)
        y_max = int(bounding_box.bottom)

        # Bounding box is in the top-right region
        cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)

    for bounding_box in bounding_boxes_time:
        x_min = int(bounding_box.left)
        x_max = int(bounding_box.right)
        y_min = int(bounding_box.top)
        y_max = int(bounding_box.bottom)

        cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)

    # Close all OpenCV windows, always do this or it will annoyingly not go away
    # You can also manually quit out with ESC key.
    cv2.destroyAllWindows()

    # If we are showing the images, display the image with the selected region and bounding boxes
    # Bounding boxes in the top-right region (time) are in one color while those in the bottom left (numerical) are in another
    if show_images:
        # Display the image with the selected region and bounding boxes
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = Image.fromarray(resized_image)
        resized_image.show()

    # Return a tuple of bounding boxes in the top-right and bottom-left regions
    return (bounding_boxes_time, bounding_boxes_numbers)
