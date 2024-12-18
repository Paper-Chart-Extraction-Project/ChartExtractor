"""Extracts the non-boxed drug and fluid data."""

# Built-in imports
from typing import Dict, List, Tuple

# Internal imports
from label_clustering.cluster import Cluster
from utilities.detections import Detection


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
    """
    pass


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
