"""Consolidates all the functionality for extracting data from charts into one function."""

# Built-in imports
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

# Internal Imports
from extraction.blood_pressure_and_heart_rate import (
    extract_heart_rate_and_blood_pressure,
)
from extraction.checkboxes import extract_checkboxes
from extraction.extraction_utilities import detect_numbers
from extraction.inhaled_volatile import extract_inhaled_volatile
from extraction.intraoperative_digit_boxes import (
    extract_drug_codes,
    extract_ett_size,
    extract_surgical_timing,
)
from extraction.physiological_indicators import extract_physiological_indicators
from extraction.preoperative_postoperative_digit_boxes import (
    extract_preop_postop_digit_data,
)
from image_registration.homography import homography_transform
from label_clustering.cluster import Cluster
from label_clustering.clustering_methods import (
    cluster_kmeans,
    cluster_boxes,
    find_legend_locations,
)
from label_clustering.isolate_labels import isolate_blood_pressure_legend_bounding_boxes
from object_detection_models.ultralytics_yolov8 import UltralyticsYOLOv8
from object_detection_models.ultralytics_yolov11_pose import UltralyticsYOLOv11Pose
from object_detection_models.object_detection_model import ObjectDetectionModel
from utilities.annotations import BoundingBox
from utilities.detections import Detection
from utilities.detection_reassembly import (
    untile_detections,
    intersection_over_minimum,
    non_maximum_suppression,
)
from utilities.tiling import tile_image


CORNER_LANDMARK_NAMES: List[str] = [
    "anesthesia_start",
    "safety_checklist",
    "lateral",
    "units",
]


def label_studio_to_bboxes(
    path_to_json_data: Path,
    desired_im_width: int = 3300,
    desired_im_height: int = 2550,
) -> List[BoundingBox]:
    """
    Convert the json data from label studio to a list of BoundingBox objects
    Args:
        path_to_json_data (Path):
            Path to the json data from label studio
    Returns:
        List[BoundingBox]:
            List of BoundingBox objects
    """
    json_data: List[Dict] = json.loads(open(str(path_to_json_data)).read())
    return {
        sheet_data["data"]["image"].split("-")[-1]: [
            BoundingBox(
                category=label["value"]["rectanglelabels"][0],
                left=label["value"]["x"] / 100 * desired_im_width,
                top=label["value"]["y"] / 100 * desired_im_height,
                right=(label["value"]["x"] / 100 + label["value"]["width"] / 100)
                * desired_im_width,
                bottom=(label["value"]["y"] / 100 + label["value"]["height"] / 100)
                * desired_im_height,
            )
            for label in sheet_data["annotations"][0]["result"]
        ]
        for sheet_data in json_data
    }


def combine_dictionaries(dictionaries: List[Dict]):
    """Combines a list of dictionaries into one.

    Args:
        `dictionaries` (List[Dict]):
            A list of dictionaries to combine.

    Returns:
        A single dictionary with the contents of all the dictionaries.
    """
    combined_dictionary: Dict = dict()
    for dictionary in dictionaries:
        combined_dictionary.update(dictionary)
    return combined_dictionary


def digitize_sheet(intraop_image: Image.Image, preop_postop_image: Image.Image) -> Dict:
    """Digitizes a paper surgical record using smartphone images of the front and back.

    Args:
        `intraop_image` (Image.Image):
            A smartphone photograph of the intraoperative side of the paper
            anesthesia record.
        `preop_postop_image` (Image.Image):
            A smartphone photograph of the preoperative/postoperative side of the
            paper anesthesia record.

    Returns:
        A dictionary containing all the data from the anesthesia record.
    """
    data = dict()
    data.update(digitize_intraop_record(intraop_image))
    data.update(digitize_preop_postop_record(preop_postop_image))
    return data


def digitize_intraop_record(image: Image.Image) -> Dict:
    """Digitizes the intraoperative side of a paper anesthesia record.

    Args:
        `image` (Image.Image):
            The smartphone image of the intraoperative side of the
            paper anesthesia record.

    Returns:
        A dictionary containing all the data from the intraoperative side of
        the paper anesthesia record.
    """
    image: Image.Image = homography_intraoperative_chart(
        image, make_document_landmark_detections(image)
    )
    document_landmark_detections: List[Detection] = make_document_landmark_detections(
        image
    )
    digit_detections: List[Detection] = make_digit_detections(image)

    # extract drug code and surgical timing
    codes: Dict = {"codes": extract_drug_codes(digit_detections, *image.size)}
    times: Dict = {"timing": extract_surgical_timing(digit_detections, *image.size)}
    ett_size: Dict = {"ett_size": extract_ett_size(digit_detections, *image.size)}

    # extract inhaled volatile drugs
    time_boxes, mmhg_boxes = isolate_blood_pressure_legend_bounding_boxes(
        [det.annotation for det in document_landmark_detections], *image.size
    )
    time_clusters: List[Cluster] = cluster_boxes(
        time_boxes, cluster_kmeans, "mins", possible_nclusters=[40, 41, 42]
    )
    mmhg_clusters: List[Cluster] = cluster_boxes(
        mmhg_boxes, cluster_kmeans, "mmhg", possible_nclusters=[18, 19, 20]
    )

    legend_locations: Dict[str, Tuple[float, float]] = find_legend_locations(
        time_clusters + mmhg_clusters
    )
    inhaled_volatile: Dict = {
        "inhaled_volatile": extract_inhaled_volatile(
            digit_detections, legend_locations, document_landmark_detections
        )
    }

    # extract bp and hr
    bp_and_hr: Dict = {
        "bp_and_hr": make_bp_and_hr_detections(image, time_clusters, mmhg_clusters)
    }

    # extract physiological indicators
    physiological_indicators: Dict = {
        "physiological_indicators": extract_physiological_indicators(
            digit_detections,
            legend_locations,
            document_landmark_detections,
            *image.size
        )
    }

    # extract checkboxes
    checkboxes: Dict = {"intraoperative_checkboxes": make_intraop_checkbox_detections(image)}

    return combine_dictionaries(
        [
            codes,
            times,
            ett_size,
            inhaled_volatile,
            bp_and_hr,
            physiological_indicators,
            checkboxes,
        ]
    )


def digitize_preop_postop_record(image: Image.Image) -> Dict:
    """Digitizes the preoperative/postoperative side of a paper anesthesia record.

    Args:
        `image` (Image.Image):
            The smartphone image of the preoperative/postopeerative
            side of the paper anesthesia record.

    Returns:
        A dictionary containing all the data from the preoperative/postoperative
        side of the paper anesthesia record.
    """
    image: Image.Image = homography_preoperative_chart(
        image,
        make_document_landmark_detections(
            image, "preop_postop_document_landmark_detector.pt"
        ),
    )
    document_landmark_detections: List[Detection] = make_document_landmark_detections(
        image
    )
    digit_detections: List[Detection] = make_digit_detections(image)
    digit_data = extract_preop_postop_digit_data(digit_detections, *image.size)
    checkbox_data = {
        "preoperative_checkboxes": make_preop_postop_checkbox_detections(image)
    }
    return combine_dictionaries([digit_data, checkbox_data])


def homography_intraoperative_chart(
    image: Image.Image, intraop_document_detections: List[Detection]
):
    """Performs a homography transformation on the intraoperative side of the chart.
    
    Args:
        `image` (Image.Image):
        `intraop_document_detections` (List[Detection]):


    """
    corner_landmark_names: List[str] = [
        "anesthesia_start",
        "safety_checklist",
        "lateral",
        "units",
    ]
    dst_landmarks = label_studio_to_bboxes("intraop_document_landmarks.json")[
        "unified_intraoperative_preoperative_flowsheet_v1_1_front.png"
    ]

    dest_points = [
        bb.center
        for bb in sorted(
            list(filter(lambda x: x.category in corner_landmark_names, dst_landmarks)),
            key=lambda bb: bb.category,
        )
    ]
    src_points = [
        bb.annotation.center
        for bb in sorted(
            list(
                filter(
                    lambda x: x.annotation.category in corner_landmark_names,
                    intraop_document_detections,
                )
            ),
            key=lambda bb: bb.annotation.category,
        )
    ]
    return homography_transform(
        image,
        dest_points=dest_points,
        src_points=src_points,
        original_image_size=(3300, 2550),  # img.size
    )
