"""Provides functions for extracting drug codes."""

# Built-in imports
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Internal imports
from handwritten_digit_utils import compute_digit_distances_to_centroids
from utilities.annotations import BoundingBox
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
        `number_detections` (List[Detection]):
            A list of Detection objects of handwritten digits.
        `im_width` (int):
            The width of the image the detections were made on.
        `im_height` (int):
            The height of the image the detections were made on.

    Returns:
        A dictionary mapping each line of the code section of the
        intraoperative record to the digits written on that line.
    """
    number_detections: List[BoundingBox] = [det.annotation for det in number_detections]
    drug_centroids: Dict[str, Tuple[float, float]] = {
        key: val for (key, val) in NUMBER_BOX_CENTROIDS.items() if "code_row" in key
    }
    drug_box_values: Dict[str, int] = compute_digit_distances_to_centroids(
        number_detections, drug_centroids, im_width, im_height
    )

    drug_codes: Dict[str, str] = dict()
    for ix in range(0, 11):
        if all(
            [
                f"code_row{str(ix).zfill(2)}_col{jx}" in drug_box_values.keys()
                for jx in range(0, 3)
            ]
        ):
            drug_codes[f"drug_row_{str(ix).zfill(2)}"] = "".join(
                [
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{0}"].category,
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{1}"].category,
                    drug_box_values[f"code_row{str(ix).zfill(2)}_col{2}"].category,
                ]
            )

    return drug_codes


def extract_surgical_timing(
    number_detections: List[Detection],
    centroids: Dict[str, Tuple[float, float]],
    im_width: int,
    im_height: int,
) -> Dict[str, str]:
    number_detections: List[BoundingBox] = [det.annotation for det in number_detections]
    surgical_timing_centroids: Dict[str, Tuple[float, float]] = {
        key: val
        for (key, val) in centroids.items()
        if any(["anes" in key, "surg" in key])
    }
    surgical_timing_values: Dict[str, int] = compute_digit_distances_to_centroids(
        number_detections, surgical_timing_centroids, im_width, im_height
    )

    surgical_timing: Dict[str, str] = dict()
    prefixes: List[str] = [
        f"{sa}_{se}_{hm}"
        for (sa, se, hm) in product(["surg", "anes"], ["start", "end"], ["hr", "min"])
    ]
    for prefix in prefixes:
        tens_place_val: Optional[int] = surgical_timing_values.get(prefix + "_tens")
        ones_place_val: Optional[int] = surgical_timing_values.get(prefix + "_ones")
        if None not in [tens_place_val, ones_place_val]:
            surgical_timing[prefix] = str(tens_place_val.category) + str(
                ones_place_val.category
            )
    return surgical_timing
