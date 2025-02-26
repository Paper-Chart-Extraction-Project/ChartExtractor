"""Consolidates all the functionality for extracting data from charts into one function."""

# Built-in imports
import os
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

# Internal Imports
from ..extraction.blood_pressure_and_heart_rate import (
    extract_heart_rate_and_blood_pressure,
)
from ..extraction.checkboxes import extract_checkboxes
from ..extraction.extraction_utilities import (
    combine_dictionaries,
    detect_numbers,
    label_studio_to_bboxes,
)
from ..extraction.inhaled_volatile import extract_inhaled_volatile
from ..extraction.intraoperative_digit_boxes import (
    extract_drug_codes,
    extract_ett_size,
    extract_surgical_timing,
)
from ..extraction.physiological_indicators import extract_physiological_indicators
from ..extraction.preoperative_postoperative_digit_boxes import (
    extract_preop_postop_digit_data,
)
from ..image_registration.homography import homography_transform
from ..label_clustering.cluster import Cluster
from ..label_clustering.clustering_methods import (
    cluster_kmeans,
    cluster_boxes,
    find_legend_locations,
)
from ..label_clustering.isolate_labels import (
    isolate_blood_pressure_legend_bounding_boxes,
)
from ..object_detection_models.ultralytics_yolov8 import UltralyticsYOLOv8
from ..object_detection_models.ultralytics_yolov11_pose import UltralyticsYOLOv11Pose
from ..object_detection_models.object_detection_model import ObjectDetectionModel
from ..utilities.detections import Detection
from ..utilities.detection_reassembly import (
    untile_detections,
    intersection_over_minimum,
    non_maximum_suppression,
)
from ..utilities.read_config import read_config
from ..utilities.tiling import tile_image


CORNER_LANDMARK_NAMES: List[str] = [
    "anesthesia_start",
    "safety_checklist",
    "lateral",
    "units",
]
PATH_TO_DATA: Path = (Path(os.path.dirname(__file__)) / ".." / ".." / "data").resolve()
PATH_TO_MODELS: Path = PATH_TO_DATA / "models"
MODEL_CONFIG: Dict = read_config()


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
    checkboxes: Dict = {
        "intraoperative_checkboxes": make_intraop_checkbox_detections(image)
    }

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
            image,
            PATH_TO_MODELS / MODEL_CONFIG["preop_postop_document_landmarks"]["name"],
        ),
    )
    digit_detections: List[Detection] = make_digit_detections(image)
    digit_data = extract_preop_postop_digit_data(digit_detections, *image.size)
    checkbox_data = {
        "preoperative_checkboxes": make_preop_postop_checkbox_detections(image)
    }
    return combine_dictionaries([digit_data, checkbox_data])


def homography_intraoperative_chart(
    image: Image.Image, intraop_document_detections: List[Detection]
) -> Image.Image:
    """Performs a homography transformation on the intraoperative side of the chart.

    Args:
        `image` (Image.Image):
            The intraoperative image to transform.
        `intraop_document_detections` (List[Detection]):
            The locations of the document landmarks.

    Returns:
        An image that is warped to correct the locations of objects on the image.
    """
    corner_landmark_names: List[str] = [
        "anesthesia_start",
        "safety_checklist",
        "lateral",
        "units",
    ]
    dst_landmarks = label_studio_to_bboxes(
        str(PATH_TO_DATA / "intraop_document_landmarks.json")
    )["unified_intraoperative_preoperative_flowsheet_v1_1_front.png"]

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


def homography_preoperative_chart(
    image: Image.Image, preop_document_detections: List[Detection]
) -> Image.Image:
    """Performs a homography transformation on the preop/postop side of the chart.

    Args:
        `image` (Image.Image):
            The preoperative/postoperative image to transform.
        `preop_document_detections` (List[Detection]):
            The locations of the document landmarks.

    Returns:
        An image that is warped to correct the locations of objects on the image.
    """
    corner_landmark_names: List[str] = [
        "patient_profile",
        "weight",
        "signature",
        "disposition",
    ]
    dst_landmarks = label_studio_to_bboxes(
        str(PATH_TO_DATA / "preoperative_document_landmarks.json")
    )["unified_intraoperative_preoperative_flowsheet_v1_1_back.png"]

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
                    preop_document_detections,
                )
            ),
            key=lambda bb: bb.annotation.category,
        )
    ]
    return homography_transform(
        image,
        dest_points=dest_points,
        src_points=src_points,
        original_image_size=(3300, 2550),
    )


def make_document_landmark_detections(
    image: Image.Image,
    document_model_filepath: Path = PATH_TO_MODELS
    / MODEL_CONFIG["intraop_document_landmarks"]["name"],
) -> List[Detection]:
    """Runs the document landmark detection model to find document landmarks.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `document_model_filepath` (Path):
            The filepath to the document landmark model weights.

    Returns:
        A list of detections containing the locations of the document landmarks.
    """
    document_model: UltralyticsYOLOv8 = UltralyticsYOLOv8.from_weights_path(
        str(document_model_filepath)
    )
    tile_size_proportion: float = MODEL_CONFIG["intraop_document_landmarks"]["tile_size_proportion"]
    tile_size: int = int(min(
        image.size[0]*tile_size_proportion,
        image.size[1]*tile_size_proportion,
    ))
    tiles: List[List[Image.Image]] = tile_image(image, tile_size, tile_size, 0.5, 0.5)
    detections = [
        [document_model(tile, verbose=False) for tile in row] for row in tiles
    ]
    detections = untile_detections(detections[0], tile_size, tile_size, 0.5, 0.5)
    detections = non_maximum_suppression(
        detections, overlap_comparator=intersection_over_minimum
    )
    del document_model
    return detections


def make_digit_detections(
    image: Image.Image,
    digit_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["numbers"]["name"],
) -> List[Detection]:
    """Runs the digit detection detection model to find handwritten digits.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `document_model_filepath` (Path):
            The filepath to the digit detector model weights.

    Returns:
        A list of detections containing the locations of handwritten digits.
    """
    digit_model: UltralyticsYOLOv8 = UltralyticsYOLOv8.from_weights_path(
        str(digit_model_filepath)
    )
    tile_size_proportion: float = MODEL_CONFIG["numbers"]["tile_size_proportion"]
    tile_size = int(min(
        image.size[0]*tile_size_proportion,
        image.size[1]*tile_size_proportion,
    ))
    number_detections: List[Detection] = detect_numbers(
        image, digit_model, tile_size, tile_size, 0.5, 0.5
    )
    del digit_model
    return number_detections


def make_bp_and_hr_detections(
    image: Image.Image,
    time_clusters: List[Cluster],
    mmhg_clusters: List[Cluster],
    sys_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["systolic"]["name"],
    dia_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["diastolic"]["name"],
    hr_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["heart_rate"]["name"],
) -> Dict:
    """Finds blood pressure symbols and associates a value and timestamp to them.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `time_clusters` (List[Cluster]):
            A list of Cluster objects encoding the location of the time legend.
        `mmhg_clusters` (List[Cluster]):
            A list of Cluster objects encoding the location of the mmhg/bpm legend.
        `sys_model_filepath` (Path):
            The filepath to the systolic symbol detector.
        `dia_model_filepath` (Path):
            The filepath to the diastolic symbol detector.
        `hr_model_filepath` (Path):
            The filepath to the heart rate symbol detector.

    Returns:
        A dictionary mapping timestamps to values for systolic, diastolic, and heart rate.
    """

    def tile_predict(
        model: ObjectDetectionModel,
        image: Image.Image,
        tile_width: int,
        tile_height: int,
        horizontal_overlap_ratio: float,
        vertical_overlap_ratio: float,
    ):
        """Performs tiled prediction."""
        tiles: List[List[Image.Image]] = tile_image(
            image,
            tile_width,
            tile_height,
            horizontal_overlap_ratio,
            vertical_overlap_ratio,
        )
        tiled_detections: List[List[List[Detection]]] = [
            [model(tile, conf=0.5) for tile in row] for row in tiles
        ]
        detections: List[Detection] = untile_detections(
            tiled_detections,
            tile_width,
            tile_height,
            horizontal_overlap_ratio,
            vertical_overlap_ratio,
        )
        return detections

    sys_model = UltralyticsYOLOv11Pose.from_weights_path(str(sys_model_filepath))
    dia_model = UltralyticsYOLOv11Pose.from_weights_path(str(dia_model_filepath))
    hr_model = UltralyticsYOLOv11Pose.from_weights_path(str(hr_model_filepath))

    sys_tile_size = int(min(
        image.size[0]*MODEL_CONFIG["systolic"]["tile_size_proportion"],
        image.size[1]*MODEL_CONFIG["systolic"]["tile_size_proportion"],
    ))
    dia_tile_size = int(min(
        image.size[0]*MODEL_CONFIG["diastolic"]["tile_size_proportion"],
        image.size[1]*MODEL_CONFIG["diastolic"]["tile_size_proportion"],
    ))
    hr_tile_size = int(min(
        image.size[0]*MODEL_CONFIG["heart_rate"]["tile_size_proportion"],
        image.size[1]*MODEL_CONFIG["heart_rate"]["tile_size_proportion"],
    ))


    sys_dets: List[Detection] = tile_predict(
        sys_model, image.copy(), sys_tile_size, sys_tile_size, 0.5, 0.5
    )
    dia_dets: List[Detection] = tile_predict(
        dia_model, image.copy(), dia_tile_size, dia_tile_size, 0.5, 0.5
    )
    hr_dets: List[Detection] = tile_predict(
        hr_model, image.copy(), hr_tile_size, hr_tile_size, 0.5, 0.5
    )

    sys_dets: List[Detection] = non_maximum_suppression(
        sys_dets, 0.5, intersection_over_minimum
    )
    dia_dets: List[Detection] = non_maximum_suppression(
        dia_dets, 0.5, intersection_over_minimum
    )
    hr_dets: List[Detection] = non_maximum_suppression(
        hr_dets, 0.5, intersection_over_minimum
    )

    dets: List[Detection] = sys_dets + dia_dets + hr_dets
    bp_and_hr = extract_heart_rate_and_blood_pressure(
        dets, time_clusters, mmhg_clusters
    )

    del sys_model
    del dia_model
    del hr_model

    return bp_and_hr


def make_intraop_checkbox_detections(
    image: Image.Image,
    checkbox_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["checkboxes"]["name"],
) -> Dict:
    """Finds checkboxes on the intraoperative form, then associates a meaning to them.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `checkbox_model_filepath` (Path):
            The filepath to the checkbox detector.

    Returns:
        A dictionary mapping names of checkboxes to a "checked" or "unchecked" state.
    """
    checkbox_model = UltralyticsYOLOv8.from_weights_path(checkbox_model_filepath)
    tile_size: int = int(min(
        image.size[0]*MODEL_CONFIG["checkboxes"]["tile_size_proportion"],
        image.size[1]*MODEL_CONFIG["checkboxes"]["tile_size_proportion"],
    ))
    intraop_checkboxes = extract_checkboxes(
        image, checkbox_model, "intraoperative", tile_size, tile_size
    )
    del checkbox_model
    return intraop_checkboxes


def make_preop_postop_checkbox_detections(
    image: Image.Image,
    checkbox_model_filepath: Path = PATH_TO_MODELS / MODEL_CONFIG["checkboxes"]["name"],
):
    """Finds checkboxes on the intraoperative form, then associates a meaning to them.

    Args:
        `image` (Image.Image):
            The image to detect on.
        `checkbox_model_filepath` (Path):
            The filepath to the checkbox detector.

    Returns:
        A dictionary mapping names of checkboxes to a "checked" or "unchecked" state.
    """
    checkbox_model = UltralyticsYOLOv8.from_weights_path(checkbox_model_filepath)
    tile_size: int = int(min(
        image.size[0]*MODEL_CONFIG["checkboxes"]["tile_size_proportion"],
        image.size[1]*MODEL_CONFIG["checkboxes"]["tile_size_proportion"],
    ))
    preop_postop_checkboxes = extract_checkboxes(
        image, checkbox_model, "preoperative", 800, 800
    )
    del checkbox_model
    return preop_postop_checkboxes
