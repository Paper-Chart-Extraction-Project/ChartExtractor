"""Provides functions for extracting the physiological indicators from the chart."""

# Built-in imports
from itertools import pairwise
from typing import Dict, List, Tuple

# External imports
import numpy as np

# Internal imports
from label_clustering.cluster import Cluster
from utilities.annotations import BoundingBox
from utilities.detections import Detection


PHYSIO_LANDMARK_NAMES: List[str] = [
    "spo2",
    "etco2",
    "fio2",
    "temperature",
    "tidal_volume",
    "respiratory_rate",
    "urine_output",
    "blood_loss",
]
