"""A module which converts legend detections into (x, y) coordinates of legend entries."""

# Built-in imports
from itertools import pairwise
from typing import Dict, List, Tuple
import warnings

# Internal imports
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection

# External imports
import numpy as np
from scipy.stats import gaussian_kde


def find_legend(
    legend_detections: List[Detection],
    image_width: int,
    image_height: int,
) -> Dict[str, Tuple[float, float]]:
    """Finds the location of the legend.

    The legend has two components. The first is timing which runs across the page left to right,
    and the second is the mmhg/bpm which runs along the page top to bottom. This function
    determines the location of each part of the legend and returns a dictionary.

    Args:
        legend_detections (List[Detection]):
            The homography-corrected legend detections.
        image_width (int):
            The image's width.
        image_height (int):
            The image's height.

    Returns:
        A dictionary whose keys are the name of the legend entry ("X_mmhg" for mmhg/bpm entries and
        "Y_mins" for time entries), and whose values are the normalized location of that legend
        marking.
    """
    bboxes: List[BoundingBox] = [det.annotation for det in legend_detections]
    time_bboxes, mmhg_bboxes = __separate_mmhg_and_timing_detections(
        bboxes,
        image_height,
        image_width,
    )

    legend_locations: Dict[str, Tuple[float, float]] = dict()
    legend_locations.update(__convert_mmhg_bboxes_to_legend_locations(mmhg_bboxes))
    legend_locations.update(__convert_time_bboxes_to_legend_locations(time_bboxes))

    return legend_locations


def __separate_mmhg_and_timing_detections(
    legend_bounding_boxes: List[BoundingBox],
    image_height: int,
    image_width: int,
) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    """Separates the timing detections from the mmhg detections.

    Args:
        legend_bounding_boxes (List[Detection]):
            The homography-corrected legend detections.
        image_height (int):
            The image's height.
        image_width (int):
            The image's width.

    Returns:
        A tuple containing the (timing detections, mmhg detections).
    """
    bboxes: List[BoundingBox] = list(
        filter(
            lambda bb: 0.2 * image_height < bb.center[1] < 0.8 * image_height,
            legend_bounding_boxes,
        )
    )

    # x_loc and y_loc form the point at the top left corner of the bp and hr section.
    x_loc: int = __find_density_max([bb.left for bb in bboxes], image_width)
    y_loc: int = __find_density_max([bb.top for bb in bboxes], image_height)

    # heuristics to determine if the box is a time box or mmhg box.
    def is_time_box(box: BoundingBox):
        return abs(box.center[0] - x_loc) > abs(box.center[1] - y_loc)

    def is_mmhg_box(box: BoundingBox):
        return abs(box.center[0] - x_loc) < abs(box.center[1] - y_loc)

    time_bboxes: List[BoundingBox] = list(filter(is_time_box, bboxes))
    mmhg_bboxes: List[BoundingBox] = list(filter(is_mmhg_box, bboxes))

    # Return a tuple of bounding boxes in the top-right and bottom-left regions
    return time_bboxes, mmhg_bboxes


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


def __convert_mmhg_bboxes_to_legend_locations(
    mmhg_bounding_boxes: List[BoundingBox],
) -> Dict[str, Tuple[float, float]]:
    """Attempts to convert the mmhg bounding boxes into pixel locations of the legend.

    Args:
        mmhg_bounding_boxes (List[BoundingBox]):
            The bounding boxes that encode the mmhg/bpm locations.

    Returns:
        A dictionary mapping the names of the mmhg/bpm legend entries (X_bpm) to (x, y) coordinates
        on the image.

    Raises:
        ValueError:
            If the function cannot resolve an issue caused by there being too many detections, too
            few detections, or too many mislabeled detections.
    """
    if len(mmhg_bounding_boxes) < 19:
        raise ValueError(
            f"Legend detection found too few legend entries for mmhg: {len(mmhg_bounding_boxes)}"
        )
    if len(mmhg_bounding_boxes) > 21:
        raise ValueError(
            f"Legend detection found too many legend entries for mmhg: {len(mmhg_bounding_boxes)}"
        )

    mmhg_legend_locations: Dict[str, Tuple[float, float]] = dict()
    mmhg_bounding_boxes: List[BoundingBox] = sorted(
        mmhg_bounding_boxes, key=lambda bb: bb.center[1], reverse=True
    )
    median_y_distance: float = np.median(
        [
            (bb_0.center[1] - bb_1.center[1])
            for (bb_0, bb_1) in pairwise(mmhg_bounding_boxes)
        ]
    )

    for ix, mmhg_bbox in enumerate(mmhg_bounding_boxes):
        is_first_box: bool = ix == 0
        is_last_box: bool = ix == len(mmhg_bounding_boxes) - 1

        if is_first_box and mmhg_bounding_boxes[0].category != "30":
            warnings.warn(
                "An anomaly was detected in the mmhg bboxes. Attempting to fix."
            )
            if (
                mmhg_bounding_boxes[0].category == "40"
                and mmhg_bounding_boxes[1].category == "50"
            ):
                mmhg_legend_locations["30_mmhg"] = (
                    mmhg_bounding_boxes[0].center[0],
                    mmhg_bounding_boxes[0].center[1] + median_y_distance,
                )
                mmhg_legend_locations["40_mmhg"] = mmhg_bbox.center
                continue
            else:
                raise ValueError("Irrecoverable anomaly in mmhg bbox detection.")
        elif is_last_box and mmhg_bounding_boxes[-1].category != "220":
            warnings.warn(
                "An anomaly was detected in the mmhg bboxes. Attempting to fix."
            )
            if (
                mmhg_bounding_boxes[-1].category == "210"
                and mmhg_bounding_boxes[-2].category == "200"
            ):
                mmhg_legend_locations["210_mmhg"] = mmhg_bbox.center
                mmhg_legend_locations["220_mmhg"] = (
                    mmhg_bounding_boxes[-1].center[0],
                    mmhg_bounding_boxes[-1].center[1] - median_y_distance,
                )
                continue
            else:
                raise ValueError("Irrecoverable anomaly in mmhg bbox detection.")
        elif all(
            [
                not is_first_box,
                not is_last_box,
                int(mmhg_bbox.category) - int(mmhg_bounding_boxes[ix - 1].category)
                != 10,
            ]
        ):
            warnings.warn(
                "An anomaly was detected in the mmhg bboxes. Attempting to fix."
            )

            previous_box_category: int = int(mmhg_bounding_boxes[ix - 1].category)
            next_box_category: int = int(mmhg_bounding_boxes[ix + 1].category)
            distance_to_previous_box: float = abs(
                mmhg_bounding_boxes[ix - 1].center[1] - mmhg_bbox.center[1]
            )
            box_is_mislabeled: bool = next_box_category - previous_box_category != 20
            # If the distance to the last box is more than 10 pixels off the median, its missing.
            previous_box_is_missing: bool = distance_to_previous_box > 10
            # If the distance to the last box is less than 10 pixels off the median, its an
            # extra box.
            box_is_erroneous: bool = (
                distance_to_previous_box < (2 / 3) * median_y_distance
            )
            if box_is_erroneous:
                pass
            elif previous_box_is_missing:
                imputed_missing_box_center: Tuple[float, float] = (
                    (0.5)
                    * (mmhg_bbox.center[0] + mmhg_bounding_boxes[ix - 1].center[0]),
                    (0.5)
                    * (mmhg_bbox.center[1] + mmhg_bounding_boxes[ix - 1].center[1]),
                )
                imputed_missing_box_label: int = int(
                    0.5 * (previous_box_category + int(mmhg_bbox.category))
                )
                mmhg_legend_locations[f"{imputed_missing_box_label}_mmhg"] = (
                    imputed_missing_box_center
                )
            elif box_is_mislabeled:
                imputed_label = int(0.5 * (next_box_category + previous_box_category))
                mmhg_legend_locations[f"{imputed_label}_mmhg"] = mmhg_bbox.center
                continue
            else:
                raise ValueError("Irrecoverable anomaly in mmhg box detection.")
        mmhg_legend_locations[f"{mmhg_bbox.category}_mmhg"] = mmhg_bbox.center
    return mmhg_legend_locations


def __convert_time_bboxes_to_legend_locations(
    time_bounding_boxes: List[BoundingBox],
) -> Dict[str, Tuple[float, float]]:
    """Attempts to convert the time bounding boxes into pixel locations of the legend.

    Args:
        time_bounding_boxes (List[BoundingBox]):
            The bounding boxes that encode the time locations.

    Returns:
        A dictionary mapping the names of the time legend entries (X_mins) to (x, y) coordinates
        on the image.

    Raises:
        ValueError:
            If the function cannot resolve an issue caused by there being too many detections, too
            few detections, or too many mislabeled detections.
    """
    if len(time_bounding_boxes) < 41:
        raise ValueError(
            f"Legend detection found too few legend entries for time: {len(time_bounding_boxes)}"
        )
    if len(time_bounding_boxes) > 43:
        raise ValueError(
            f"Legend detection found too many legend entries for time: {len(time_bounding_boxes)}"
        )

    time_legend_locations: Dict[str, Tuple[float, float]] = dict()
    time_bounding_boxes: List[BoundingBox] = sorted(
        time_bounding_boxes,
        key=lambda bb: bb.center[0],
    )
    median_x_distance: float = np.median(
        [
            (bb_1.center[0] - bb_0.center[0])
            for (bb_0, bb_1) in pairwise(time_bounding_boxes)
        ]
    )

    def timedelta(ix: int):
        return (ix // 12) * 60

    for ix, time_bbox in enumerate(time_bounding_boxes):
        is_first_box: bool = ix == 0
        is_last_box: bool = ix == len(time_bounding_boxes) - 1
        time_gap_too_large = (int(time_bbox.category) + timedelta(ix)) - (
            int(time_bounding_boxes[ix - 1].category) + timedelta(ix - 1)
        ) != 5  # There should only be 5 minutes between legend entries.
        if is_first_box and time_bounding_boxes[0].category != "0":
            warnings.warn(
                "An anomaly was detected in the time bboxes. Attempting to fix."
            )
            if (
                time_bounding_boxes[0].category == "5"
                and time_bounding_boxes[1].category == "10"
            ):
                time_legend_locations["0_mins"] = (
                    time_bounding_boxes[0].center[0] - median_x_distance,
                    time_bounding_boxes[0].center[1],
                )
                time_legend_locations["5_mins"] = time_bbox.center
                continue
            else:
                raise ValueError("Irrecoverable anomaly in time bbox detection.")
        elif is_last_box and time_bounding_boxes[-1].category != "25":
            warnings.warn(
                "An anomaly was detected in the time bboxes. Attempting to fix."
            )
            if (
                time_bounding_boxes[-1].category == "20"
                and time_bounding_boxes[-2].category == "15"
            ):
                time_legend_locations["200_mins"] = time_bbox.center
                time_legend_locations["205_mins"] = (
                    time_bounding_boxes[-1].center[0] + median_x_distance,
                    time_bounding_boxes[-1].center[1],
                )
                continue
            else:
                raise ValueError("Irrecoverable anomaly in time bbox detection.")
        elif all([not is_first_box, not is_last_box, time_gap_too_large]):
            warnings.warn(
                "An anomaly was detected in the time bboxes. Attempting to fix."
            )

            previous_box_category: int = int(
                time_bounding_boxes[ix - 1].category
            ) + timedelta(ix - 1)
            next_box_category: int = int(
                time_bounding_boxes[ix + 1].category
            ) + timedelta(ix + 1)
            distance_to_previous_box: float = abs(
                time_bounding_boxes[ix - 1].center[0] - time_bbox.center[0]
            )
            box_is_mislabeled: bool = next_box_category - previous_box_category != 10
            # If the distance to the last box is more than 10 pixels off the median, its missing.
            previous_box_is_missing: bool = (
                distance_to_previous_box - median_x_distance > 10
            )
            # If the distance to the last box is less than 10 pixels off the median, its an
            # extra box.
            box_is_erroneous: bool = (
                distance_to_previous_box < (2 / 3) * median_x_distance
            )
            if box_is_erroneous:
                pass
            elif previous_box_is_missing:
                imputed_missing_box_center: Tuple[float, float] = (
                    (0.5)
                    * (time_bbox.center[0] + time_bounding_boxes[ix - 1].center[0]),
                    (0.5)
                    * (time_bbox.center[1] + time_bounding_boxes[ix - 1].center[1]),
                )
                imputed_missing_box_label: int = int(
                    0.5
                    * (previous_box_category + int(time_bbox.category) + timedelta(ix))
                )
                time_legend_locations[
                    f"{imputed_missing_box_label+timedelta(ix)}_mins"
                ] = imputed_missing_box_center
            elif box_is_mislabeled:
                imputed_label = int(0.5 * (next_box_category + previous_box_category))
                time_legend_locations[f"{imputed_label+timedelta(ix)}_mins"] = (
                    time_bbox.center
                )
                continue
            else:
                raise ValueError("Irrecoverable anomaly in time box detection.")
        time_legend_locations[f"{int(time_bbox.category)+timedelta(ix)}_mins"] = (
            time_bbox.center
        )

    return time_legend_locations
