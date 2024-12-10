"""Provides functions for extracting drug codes."""

# Built-in imports
from pathlib import Path
from typing import Dict, List, Tuple

# Internal imports
from utilities.detections import Detection


FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = Path("")
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = {}


def extract_drug_codes(
    number_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, int]:
    """ """
    pass
