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
    """ """
    pass
