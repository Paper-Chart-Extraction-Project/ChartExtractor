"""Provides functions for extracting drug codes."""

# Built-in imports
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Internal imports
from utilities.detections import Detection


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_single_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)


def extract_drug_codes(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, int]:
    """Extracts the drug code data from the number detections.

    Args:
        number_detections (List[Detection]):
            A list of Detection objects of handwritten digits.
        im_width (int):
            The width of the image the detections were made on.
        im_height (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping each line of the code section of the
        intraoperative record to the digits written on that line.
    """
    pass
