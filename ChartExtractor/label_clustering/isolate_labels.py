"""Module that contains code for isolating labels from a list of bounding boxes from the blood pressure section.

This module contains functions for isolating labels from a list of bounding boxes.
It includes a public function for external use and several worker functions that do the isolation.
"""

# Built-in imports
from operator import attrgetter
from typing import List, Tuple

# External imports
import numpy as np
from scipy.stats import gaussian_kde

# Internal imports
from ..utilities.annotations import BoundingBox


def __find_density_max(values: List[int], search_area: int) -> int:
    """Given a list of values and a search area, find the index of where the highest density is.

    The list of values correspond to identifying points for the bounding boxes and the search
    area corresponds to the images height or width.

    Args:
        `values` (List[int]):
            List of identifying points for the bounding boxes
        `search_area` (int):
            height/width of the image dependent on whether x or y axis is being searched.

    Returns:
        The axis value that has the highest density of bounding boxes.
    """
    kde = gaussian_kde(values, bw_method=0.2)
    values = np.linspace(0, search_area, 10000)
    kde_vals = kde(values)
    max_index = np.argmax(kde_vals)
    return values[max_index]


def __remove_bb_outliers(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Given a list of bounding boxes, remove the outliers.

    Removes outliers from the x axis, then remove the outliers from the y axis.

    Args:
        `boxes` (List[BoundingBox]):
            List of Bounding Boxes to filter

    Returns:
        Filtered list of Bounding Boxes
    """

    def filter_outlier_values(
        boxes: List[BoundingBox], box_side: str
    ) -> List[BoundingBox]:
        """Filters values more than 1.5 IQRs away from the 25th and 75th percentiles.

        Does this for both x and y.
        """
        get_box_side = attrgetter(box_side)
        values: List[float] = list(map(get_box_side, boxes))
        first_quartile, third_quartile = np.percentile(values, [25, 75])
        interquartile_range: float = third_quartile - first_quartile
        bounds: Tuple[float, float] = (
            first_quartile - 1.5 * interquartile_range,
            third_quartile + 1.5 * interquartile_range,
        )
        return list(
            filter(lambda box: bounds[0] <= get_box_side(box) <= bounds[1], boxes)
        )

    filtered = filter_outlier_values(boxes, "left")
    filtered = filter_outlier_values(boxes, "top")
    return filtered


def isolate_blood_pressure_legend_bounding_boxes(
    document_landmark_boxes: List[BoundingBox],
    im_width: int = 800,
    im_height: int = 600,
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """Isolates the bounding boxes that form the blood pressure legend.

    Given bounding boxes of document landmarks that include printed digits, finds the
    bounding boxes corresponding to the mmhg/bpm and time legend.

    Args:
        `document_landmark_boxes` (List[BoundingBox]):
            List of bounding boxes encoding the location of all document landmarks.
        `im_width` (int):
            Width of the image. Default is 800.
        `im_height` (int):
            Height of the image. Default is 600.

    Returns:
        Two lists of bounding boxes.
        The first list contains bounding boxes in the top-right representing time labels.
        The second list contains bounding boxes in the bottom-left representing values
        for mmHg and bpm.
        (time_bboxes, mmhg_bboxes)
    """
    # filter out bounding boxes whose category is not a digit, and which are certainly not
    # in the region of interest.
    bboxes: List[BoundingBox] = list(
        filter(
            lambda bb: all(
                [
                    bb.category in [str(i) for i in range(10)],
                    0.2 * im_height < bb.center[1] < 0.8 * im_height,
                ]
            ),
            document_landmark_boxes,
        )
    )

    # x_loc and y_loc form the point at the top left corner of the bp and hr section.
    x_loc: int = __find_density_max([bb.left for bb in bboxes], im_width)
    y_loc: int = __find_density_max([bb.top for bb in bboxes], im_height)

    # heuristics to determine if the box is a time box or mmhg box.
    is_time_box = lambda box: abs(box.center[0] - x_loc) > abs(box.center[1] - y_loc)
    is_mmhg_box = lambda box: abs(box.center[0] - x_loc) < abs(box.center[1] - y_loc)

    time_bboxes: List[BoundingBox] = list(filter(is_time_box, bboxes))
    mmhg_bboxes: List[BoundingBox] = list(filter(is_mmhg_box, bboxes))

    # Return a tuple of bounding boxes in the top-right and bottom-left regions
    return time_bboxes, mmhg_bboxes
