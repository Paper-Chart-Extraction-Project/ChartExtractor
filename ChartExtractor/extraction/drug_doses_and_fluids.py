"""Extracts the non-boxed drug and fluid data."""

# Built-in imports
from functools import partial
import json
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Internal imports
from ..extraction.extraction_utilities import average_with_nones, get_detection_by_name
from ..label_clustering.cluster import Cluster
from ..utilities.detections import Detection

# External imports
import numpy as np


DATA_FILEPATH: Path = Path(__file__).parents[2] / "data"
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    DATA_FILEPATH / "centroids" / "intraop_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)


def extract_drug_dosages_and_fluids(
    digit_detections: List[Detection],
    legend_locations: Dict[str, Tuple[float, float]],
    document_detections: List[Detection],
    im_width: int,
    im_height: int,
) -> Dict[str, Dict[str, str]]:
    """Extracts all the drug dosages and fluid from the digit detections.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten digits that have been detected.
        `legend_locations` (Dict[str, Tuple[float, float]]):
            The location of timestamps and mmhg/bpm values on the legend.
        `document_detections` (List[Detection]):
            The printed document landmarks that have been detected.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping the row to a dictionary that maps timestamps to dosages/fluid amounts.
    """
    pass


def get_drug_dosage_digits(
    digit_detections: List[Detection],
    document_detections: List[Detection],
) -> List[Detection]:
    """Filters for the digit detections that are within the drug dosage section.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `document_detections` (List[Detection]):
            The location of document landmarks that have been detected.

    Returns:
        A filtered list of detections holding only those that are in the drug dosage section.

    Raises:
        ValueError:
            If any of the necessary document detections cannot be found.
    """
    get_detection_by_name = partial(
        get_detection_by_name, detections=document_detections
    )
    drug_name: Optional[Detection] = get_detection_by_name("drug_name")
    units: Optional[Detection] = get_detection_by_name("units")
    inhaled_volatile: Optional[Detection] = get_detection_by_name("inhaled_volatile")
    inhaled_exhaled: Optional[Detection] = get_detection_by_name("inhaled_exhaled")

    if any(
        [
            drug_name is None,
            units is None,
            inhaled_volatile is None,
            inhaled_exhaled is None,
        ]
    ):
        raise ValueError("Cannot find all necessary document detections.")

    left: float = np.mean(
        list(map([drug_name, inhaled_volatile], attrgetter("annotation")))
    )
    top: float = np.mean(list(map[drug_name, units]), attrgetter("annotation"))
    right: float = np.mean(list(map[units, inhaled_exhaled]), attrgetter("annotation"))
    bottom: float = np.mean(
        list(map[inhaled_volatile, inhaled_exhaled]), attrgetter("annotation")
    )

    def detection_is_in_region(detection: Detection) -> bool:
        center = attrgetter("annotation.center")
        (left < center(detection)[0] < right) and (top < center(detection)[1] < bottom)

    return list(filter(detection_is_in_region, digit_detections))


def get_fluid_digits(
    digit_detections: List[Detection],
    document_detections: List[Detection],
) -> List[Detection]:
    """Filters for the digit detections that are within the fluid section.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten numbers that have been detected.
        `document_detections` (List[Detection]):
            The location of document landmarks that have been detected.

    Returns:
        A filtered list of detections holding only those that are in the fluid section.
    """
    pass


def cluster_digits(
    digit_detections: List[Detection],
) -> List[Cluster]:
    """Clusters the digits using KMeans.

    Args:
        `digit_detections` (List[Detection]):
            The handwritten digits which have been detected on the sheet.

    Returns:
        A list of Cluster objects grouping the digits.
    """
    pass
