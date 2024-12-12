"""Provides functions for extracting preoperative/postoperative handwritten digit data."""

# Built-in imports
from itertools import product
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Internal imports
from extraction.handwritten_digit_utils import compute_digit_distances_to_centroids
from utilities.annotations import BoundingBox
from utilities.detections import Detection


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "preop_postop_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)
