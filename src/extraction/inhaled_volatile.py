"""Extracts the inhaled volatile drug data."""

# Built-in imports
from operator import attrgetter
from functools import reduce
from typing import Dict, List, Optional, Tuple

# Internal imports
from utilities.detections import Detection


def extract_inhaled_volatile(
    digit_detections: List[Detection],
    legend_locations: Dict[str, Tuple[float, float]],
    document_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    """Extracts the inhaled volatile gas data from the number detections.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `legend_locations` (Dict[str, Tuple[float, float]]):
            The location of the timestamps and mmhg/bpm values on the legend.
        `document_detections` (List[Detection]):
            The location of document landmarks that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping timestamps to inhaled volatile gas data.
    """
    inhaled_volatile_digits: List[Detection] = get_inhaled_volatile_digits(
        digit_detections,
        document_detections,
    )


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

    def get_detection_by_name(name: str) -> Optional[Detection]:
        try:
            return list(
                filter(lambda d: d.annotation.category == name, document_detections)
            )[0]
        except:
            return None

    def average_with_nones(list_with_nones: List[Optional[float]]) -> float:
        add_with_none = lambda acc, x: acc + x if x is not None else acc
        len_with_none = lambda l: len(list(filter(lambda x: x is not None, l)))
        return reduce(add_with_none, list_with_nones) / len_with_none(list_with_nones)

    inhaled_volatile: Optional[Detection] = get_detection_by_name("inhaled_volatile")
    inhaled_exhaled: Optional[Detection] = get_detection_by_name("inhaled_exhaled")
    fluid_blood_product: Optional[Detection] = get_detection_by_name(
        "fluid_blood_product"
    )
    total: Optional[Detection] = get_detection_by_name("total")

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
