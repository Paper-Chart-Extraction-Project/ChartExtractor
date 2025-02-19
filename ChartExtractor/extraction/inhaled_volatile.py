"""Extracts the inhaled volatile drug data."""

# Built-in imports
from itertools import pairwise
from operator import attrgetter
from typing import Dict, List, Optional, Tuple

# Internal imports
from .extraction.extraction_utilities import average_with_nones, get_detection_by_name
from ..utilities.annotations import BoundingBox
from ..utilities.detections import Detection


def extract_inhaled_volatile(
    digit_detections: List[Detection],
    legend_locations: Dict[str, Tuple[float, float]],
    document_detections: List[Detection],
) -> Dict[str, str]:
    """Extracts the inhaled volatile gas data from the number detections.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `legend_locations` (Dict[str, Tuple[float, float]]):
            The location of the timestamps and mmhg/bpm values on the legend.
        `document_detections` (List[Detection]):
            The location of document landmarks that have been detected.

    Returns:
        A dictionary mapping timestamps to inhaled volatile gas data.
    """
    inhaled_volatile_digits: List[Detection] = get_inhaled_volatile_digits(
        digit_detections,
        document_detections,
    )
    inhaled_volatile_digits: List[BoundingBox] = list(
        map(attrgetter("annotation"), inhaled_volatile_digits)
    )
    fifteen_minute_intervals: Dict[str, Tuple[float, float]] = {
        k: v
        for (k, v) in legend_locations.items()
        if int(k.split("_")[0]) % 15 == 0 and "mins" in k
    }
    key_pairs: List[Tuple[str, str]] = list(
        pairwise(
            sorted(
                list(fifteen_minute_intervals.keys()),
                key=lambda s: int(s.split("_")[0]),
            )
        )
    )
    inhaled_volatile: Dict[str, str] = dict()
    for pair in key_pairs:
        left: float = fifteen_minute_intervals[pair[0]][0]
        right: float = fifteen_minute_intervals[pair[1]][0]
        boxes_in_range: List[BoundingBox] = sorted(
            list(
                filter(lambda bb: left < bb.center[0] < right, inhaled_volatile_digits)
            ),
            key=lambda bb: bb.center[0],
        )
        if len(boxes_in_range) == 1:
            inhaled_volatile[f"{pair[0]}-{pair[1]}"] = f"0.{boxes_in_range[0].category}"
        elif len(boxes_in_range) == 2:
            inhaled_volatile[f"{pair[0]}-{pair[1]}"] = (
                f"{boxes_in_range[0].category}.{boxes_in_range[1].category}"
            )
        else:
            pass

    return inhaled_volatile


def get_inhaled_volatile_digits(
    digit_detections: List[Detection], document_detections: List[Detection]
) -> List[Detection]:
    """Filters for the digit detections that are within the inhaled volatile section.

    args:
        `digit_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `document_detections` (List[Detection]):
            The location of document landmarks that have been detected.

    Returns:
        A filtered list of detections holding only those that are in the inhaled volatile section.
    """

    inhaled_volatile: Optional[Detection] = get_detection_by_name(
        document_detections, "inhaled_volatile"
    )
    inhaled_exhaled: Optional[Detection] = get_detection_by_name(
        document_detections, "inhaled_exhaled"
    )
    fluid_blood_product: Optional[Detection] = get_detection_by_name(
        document_detections, "fluid_blood_product"
    )
    total: Optional[Detection] = get_detection_by_name(document_detections, "total")

    if any(
        [
            all([inhaled_volatile is None, inhaled_exhaled is None]),
            all([fluid_blood_product is None, total is None]),
        ]
    ):
        raise ValueError("Cannot find document landmarks to filter digits.")

    get_center = attrgetter("annotation.center")
    top: float = average_with_nones(
        [get_center(inhaled_volatile)[1], get_center(inhaled_exhaled)[1]]
    )
    bottom: float = average_with_nones(
        [get_center(fluid_blood_product)[1], get_center(total)[1]]
    )

    return list(filter(lambda det: top < get_center(det)[1] < bottom, digit_detections))
