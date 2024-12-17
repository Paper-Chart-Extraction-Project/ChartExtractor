"""Extracts the inhaled volatile drug data."""

# Built-in imports
from typing import Dict, List, Tuple

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
    pass
