"""Extracts the non-boxed drug and fluid data."""

# Built-in imports
from functools import partial
from itertools import product
import json
from operator import attrgetter
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeAlias

# Internal Imports
from ChartExtractor.extraction.extraction_utilities import get_detection_by_name
from ChartExtractor.label_clustering.cluster import Cluster
from ChartExtractor.utilities.annotations import BoundingBox
from ChartExtractor.utilities.detections import Detection

# External Imports
import networkx as nx
import numpy as np
import pandas as pd


PATH_TO_DATA: Path = (Path(__file__) / ".." / ".." / "data").resolve()
FILEPATH_TO_NUMBER_BOX_CENTROIDS: Path = (
    PATH_TO_DATA / "centroids" / "intraop_digit_box_centroids.json"
)
NUMBER_BOX_CENTROIDS: Dict[str, Tuple[float, float]] = json.load(
    open(FILEPATH_TO_NUMBER_BOX_CENTROIDS, "r")
)


RelativeBoundingBox: TypeAlias = BoundingBox


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
    drug_dosage_digit_detections: List[Detection] = get_drug_dosage_digits(
        digit_detections,
        document_detections,
    )
    fluid_digit_detections: List[Detection] = get_fluid_digits(
        digit_detections,
        legend_locations,
        document_detections,
    )

    drug_dosage_relative_bboxes: List[RelativeBoundingBox] = [
        convert_boundingbox_to_relativeboundingbox(
            det.annotation, # type: ignore
            im_width,
            im_height,
        )
        for det in drug_dosage_digit_detections
    ]
    fluid_relative_bboxes: List[RelativeBoundingBox] = [
        convert_boundingbox_to_relativeboundingbox(
            det.annotation, # type: ignore
            im_width,
            im_height
        )
        for det in fluid_digit_detections
    ]

    model = load_model()
    prediction_fn: Callable[[Tuple[RelativeBoundingBox, RelativeBoundingBox]], bool] = (
        partial(sklearn_prediction_fn, model=model, confidence_threshold=0.57)
    )

    drug_dosage_clusters: List[Cluster] = [
        Cluster(
            box_list,
            "".join(
                [bbox.category for bbox in sorted(box_list, key=attrgetter("left"))]
            ),
        )
        for box_list in predict_clusters(
            drug_dosage_relative_bboxes,
            euclidean_distance_proposal_fn,
            prediction_fn,
        )
    ]
    fluid_clusters: List[Cluster] = [
        Cluster(
            box_list,
            "".join(
                [bbox.category for bbox in sorted(box_list, key=attrgetter("left"))]
            ),
        )
        for box_list in predict_clusters(
            fluid_relative_bboxes,
            euclidean_distance_proposal_fn,
            prediction_fn,
        )
    ]
    
    timestamp_legend_relative_locations: Dict[str, Tuple[float, float]] = {
        ts_name: (ts_val[0]/im_width, ts_val[1]/im_height)
        for (ts_name, ts_val) in legend_locations.items()
        if "_mins" in ts_name
    }

    code_row_relative_locations: Dict[str, Tuple[float, float]] = {
        code_row_name: code_row_location
        for (code_row_name, code_row_location) in NUMBER_BOX_CENTROIDS.items()
        if "code_row" in code_row_name
    }

    drug_dosages_and_fluids: Dict[str, Dict[str, str]] = dict()
    for cluster in drug_dosage_clusters + fluid_clusters:
        row: str = find_row_for_cluster(cluster, code_row_relative_locations)
        timestamp: str = find_timestamp_for_cluster(cluster, timestamp_legend_relative_locations)
        if drug_dosages_and_fluids.get(row) is None:
            drug_dosages_and_fluids[row] = {timestamp: cluster.label}
        else:
            drug_dosages_and_fluids[row][timestamp] = cluster.label

    return drug_dosages_and_fluids


def find_timestamp_for_cluster(
    cluster: Cluster, timestamp_legend_locations: Dict[str, Tuple[float, float]]
) -> str:
    """Finds the closest timestamp for the cluster.

    Args:
        cluster (Cluster):
            The cluster to find the timestamp for.
        timestamp_legend_locations (Dict[str, Tuple[float, float]]):
            The legend locations for just the timestamps.

    Returns:
        The name of the closest timestamp.
    """
    cluster_center_x: float = cluster.bounding_box.center[0]
    distance_dict: Dict[str, float] = {
        timestamp_name: abs(cluster_center_x - timestamp_location[0])
        for (timestamp_name, timestamp_location) in timestamp_legend_locations.items()
    }
    return min(distance_dict, key=distance_dict.get)  # type: ignore


def find_row_for_cluster(
    cluster: Cluster,
    code_row_locations: Dict[str, Tuple[float, float]]
) -> str:
    """Finds the closest row for the cluster.

    Args:
        cluster (Cluster):
            The cluster to find the timestamp for.

    Returns:
        The row name for the cluster (first row is 'row_00', the last is 'row_10').
    """
    code_row_avg_y_locations: Dict[str, float] = dict()
    number_of_rows: int = 10
    for row_num in range(number_of_rows + 1):
        first_col_y: float = code_row_locations[f"code_row{row_num:02}_col0"][1]
        second_col_y: float = code_row_locations[f"code_row{row_num:02}_col1"][1]
        third_col_y: float = code_row_locations[f"code_row{row_num:02}_col2"][1]

        average_y: float = (first_col_y + second_col_y + third_col_y) / 3

        code_row_avg_y_locations[f"code_row{row_num:02}"] = average_y

    cluster_center_y: float = cluster.bounding_box.center[1]
    distance_dict: Dict[str, float] = {
        code_row_name: abs(cluster_center_y - code_row_y_location)
        for (code_row_name, code_row_y_location) in code_row_avg_y_locations.items()
    }
    return min(distance_dict, key=distance_dict.get)  # type: ignore


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
    get_det_by_name = partial(get_detection_by_name, detections=document_detections)
    drug_name: Optional[Detection] = get_det_by_name(name="drug_name")
    units: Optional[Detection] = get_det_by_name(name="units")
    inhaled_volatile: Optional[Detection] = get_det_by_name(name="inhaled_volatile")
    inhaled_exhaled: Optional[Detection] = get_det_by_name(name="inhaled_exhaled")

    any_required_detection_not_found: bool = any(
        [d is None for d in [drug_name, units, inhaled_exhaled, inhaled_volatile]]
    )
    if any_required_detection_not_found:
        raise ValueError("Cannot find all necessary document detections.")

    left: float = float(
        np.mean(list(map(attrgetter("annotation.right"), [drug_name, inhaled_volatile])))
    )
    top: float = float(
        np.mean(list(map(attrgetter("annotation.bottom"), [drug_name, units])))
    )
    right: float = inhaled_exhaled.annotation.left # type: ignore
    bottom: float = float(
        np.mean(
            list(
                map(
                    attrgetter("annotation.top"), [inhaled_volatile, inhaled_exhaled]
                )
            )
        )
    )

    def detection_is_in_region(detection: Detection) -> bool:
        center = attrgetter("annotation.center")
        return (left < center(detection)[0] < right) and (
            top < center(detection)[1] < bottom
        )

    return list(filter(detection_is_in_region, digit_detections))


def get_fluid_digits(
    digit_detections: List[Detection],
    legend_locations: Dict[str, Tuple[float, float]],
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
    get_det_by_name = partial(get_detection_by_name, detections=document_detections)
    fluid_blood_product: Optional[Detection] = get_det_by_name(name="fluid_blood_product")
    total: Optional[Detection] = get_det_by_name(name="total")
    zero_mins: Optional[Tuple[float, float]] = legend_locations.get("0_mins")
    twohundred_five_mins: Optional[Tuple[float, float]] = legend_locations.get(
        "205_mins"
    )

    any_required_detection_not_found: bool = any(
        [
            d is None
            for d in [fluid_blood_product, total, zero_mins, twohundred_five_mins]
        ]
    )
    if any_required_detection_not_found:
        raise ValueError("Cannot find all necessary document detections.")

    left: float = float(np.mean([fluid_blood_product.annotation.right, zero_mins[0]]))  # type: ignore
    top: float = float(
        np.mean(list(map(attrgetter("annotation.top"), [fluid_blood_product, total])))
    )
    right: float = float(np.mean([total.annotation.left, twohundred_five_mins[0]]))  # type: ignore
    bottom: float = float(np.mean([zero_mins[1], twohundred_five_mins[1]]))  # type: ignore

    def detection_is_in_region(detection: Detection) -> bool:
        center = attrgetter("annotation.center")
        return (left < center(detection)[0] < right) and (
            top < center(detection)[1] < bottom
        )

    return list(filter(detection_is_in_region, digit_detections))


def load_model(filename: str = "clustering_model.pkl"):
    """Loads the sklearn RandomForestClassifier that predicts single linkage of boxes.

    Loads the model from a pickle file.

    Args:
        filepath (str):
            The filename of the model.
    """
    with open(str(PATH_TO_DATA / "models" / filename), "rb") as f:
        model = pickle.load(f)
    return model


def convert_boundingbox_to_relativeboundingbox(
    bounding_box: BoundingBox,
    image_width: float,
    image_height: float,
) -> RelativeBoundingBox:
    """Converts a BoundingBox to a RelativeBoundingBox which uses relative coordinates.

    Args:
        bounding_box (BoundingBox):
            The bounding box to convert.
        image_width (int):
            The width of the image.
        image_height (int):
            The height of the image.

    Returns:
        A RelativeBoundingBox (same type as a bounding box) which is a BoundingBox that uses
        relative coordinates.
    """
    return BoundingBox(
        category=bounding_box.category,
        left=bounding_box.left / image_width,
        top=bounding_box.top / image_height,
        right=bounding_box.right / image_width,
        bottom=bounding_box.bottom / image_height,
    )


def euclidean_distance(
    rel_bbox_1: RelativeBoundingBox,
    rel_bbox_2: RelativeBoundingBox,
) -> float:
    """The euclidean distance between the two bounding boxes.

    Args:
        rel_bbox_1 (RelativeBoundingBox):
            The first relative bounding box.
        rel_bbox_2 (RelativeBoundingBox):
            The second relative bounding box.

    Returns:
        The euclidean distance between the center of the two bounding boxes.
    """
    rel_bbox_1_relative_x: float = rel_bbox_1.center[0]
    rel_bbox_1_relative_y: float = rel_bbox_1.center[1]

    rel_bbox_2_relative_x: float = rel_bbox_2.center[0]
    rel_bbox_2_relative_y: float = rel_bbox_2.center[1]

    return (rel_bbox_1_relative_x - rel_bbox_2_relative_x) ** 2 + (
        rel_bbox_1_relative_y - rel_bbox_2_relative_y
    ) ** 2


def manhattan_distance(
    rel_bbox_1: RelativeBoundingBox,
    rel_bbox_2: RelativeBoundingBox,
) -> float:
    """The manhattan distance between the two bounding boxes.

    Args:
        rel_bbox_1 (RelativeBoundingBox):
            The first relative bounding box.
        rel_bbox_2 (RelativeBoundingBox):
            The second relative bounding box.

    Returns:
        The manhattan distance between the center of the two bounding boxes.
    """
    rel_bbox_1_relative_x: float = rel_bbox_1.center[0]
    rel_bbox_1_relative_y: float = rel_bbox_1.center[1]

    rel_bbox_2_relative_x: float = rel_bbox_2.center[0]
    rel_bbox_2_relative_y: float = rel_bbox_2.center[1]

    return abs(rel_bbox_1_relative_x - rel_bbox_2_relative_x) + abs(
        rel_bbox_1_relative_y - rel_bbox_2_relative_y
    )


def euclidean_distance_proposal_fn(
    relative_bounding_boxes: List[RelativeBoundingBox],
    max_distance: float = 0.03,
) -> List[Tuple[int, int]]:
    """A proposal function which proposes via euclidean distance.

    Args:
        relative_bounding_boxes (List[RelativeBoundingBox]):
            The relative bounding boxes to create proposals for.
        max_distance (float):
            The maximum euclidean distance below which two relative bounding boxes are proposed.

    Returns:
        A list of tuples, each of which contains the indices of two boxes in the input list which
        the linkage model needs to run on to determine if they are linked or not.
    """
    proposals: List[Tuple[int, int]] = list()
    index_pairs_to_check: List[Tuple[int, int]] = list(
        product(
            range(len(relative_bounding_boxes)), range(len(relative_bounding_boxes))
        )
    )

    for rel_bbox_ix_1, rel_bbox_ix_2 in index_pairs_to_check:
        rel_bbox_1: RelativeBoundingBox = relative_bounding_boxes[rel_bbox_ix_1]
        rel_bbox_2: RelativeBoundingBox = relative_bounding_boxes[rel_bbox_ix_2]
        distance: float = euclidean_distance(rel_bbox_1, rel_bbox_2)
        if distance < max_distance:
            proposals.append((rel_bbox_ix_1, rel_bbox_ix_2))
    return proposals


def sklearn_prediction_fn(
    relative_bounding_boxes: Tuple[RelativeBoundingBox, RelativeBoundingBox],
    model,
    confidence_threshold: float,
) -> bool:
    """A prediction function which predicts whether two annotations are linked.

    Args:
        relative_bounding_boxes (Tuple[RelativeBoundingBox, RelativeBoundingBox]):
            The relative bounding boxes to determine are linked or not.
        model:
            Any sklearn classification model that can call predict_proba.
        confidence_threshold (float):
            The threshold below which a predicted probability evaluates to False.

    Returns:
        Whether or not the model believes the relative bounding boxes are linked.
    """
    rel_bbox_1: RelativeBoundingBox = min(
        relative_bounding_boxes, key=lambda rel_bb: rel_bb.center[0]
    )
    rel_bbox_2: RelativeBoundingBox = max(
        relative_bounding_boxes, key=lambda rel_bb: rel_bb.center[0]
    )

    rel_bbox_1_x: float = rel_bbox_1.center[0]
    rel_bbox_1_y: float = rel_bbox_1.center[1]
    rel_bbox_2_x: float = rel_bbox_2.center[0]
    rel_bbox_2_y: float = rel_bbox_2.center[1]

    rel_bbox_1_width: float = rel_bbox_1.right - rel_bbox_1.left
    rel_bbox_1_height: float = rel_bbox_1.bottom - rel_bbox_1.top
    rel_bbox_2_width: float = rel_bbox_2.right - rel_bbox_2.left
    rel_bbox_2_height: float = rel_bbox_2.bottom - rel_bbox_2.top

    model_input_dict: Dict[str, List[Any]] = {
        "annotation_1_category": [rel_bbox_1.category],
        "annotation_1_relative_x": [rel_bbox_1_x],
        "annotation_1_relative_y": [rel_bbox_1_y],
        "annotation_1_relative_width": [rel_bbox_1_width],
        "annotation_1_relative_height": [rel_bbox_1_height],
        "annotation_2_category": [rel_bbox_2.category],
        "annotation_2_relative_x": [rel_bbox_2_x],
        "annotation_2_relative_y": [rel_bbox_2_y],
        "annotation_2_relative_width": [rel_bbox_2_width],
        "annotation_2_relative_height": [rel_bbox_2_height],
        "euclidean_distance": [euclidean_distance(rel_bbox_1, rel_bbox_2)],
        "manhattan_distance": [manhattan_distance(rel_bbox_1, rel_bbox_2)],
        "x_diff": [rel_bbox_1_x - rel_bbox_2_x],
        "y_diff": [rel_bbox_1_y - rel_bbox_2_y],
        "right_to_left_side_diff": [rel_bbox_1.right - rel_bbox_2.left],
    }
    model_input: pd.DataFrame = pd.DataFrame(
        model_input_dict,
        columns=model.feature_names_in_,  # type: ignore
    )
    output: Tuple[float, float] = model.predict_proba(model_input).tolist()[0]
    return output[1] >= confidence_threshold


def predict_clusters(
    relative_bounding_boxes: List[RelativeBoundingBox],
    proposal_fn: Callable[[List[RelativeBoundingBox]], List[Tuple[int, int]]],
    prediction_fn: Callable[[Tuple[RelativeBoundingBox, RelativeBoundingBox]], bool],
) -> List[List[RelativeBoundingBox]]:
    """Predicts the clusters of the boxes.

    Args:
        relative_bounding_boxes (List[RelativeBoundingBox]):
            The relative bounding boxes to cluster.
        proposal_fn (Callable[[List[RelativeBoundingBox], List[Tuple[int, int]]]]):
            A function which takes a list of relative bounding boxes, and returns a list of index
            pairs that encode a proposed linkage.
        prediction_fn (Callable[[Tuple[RelativeBoundingBox, RelativeBoundingBox]], bool]):
            A function which takes a pair of annotations and predicts whether or not they are
            linked.

    Returns:
        A list of clustered relative bounding boxes.
    """
    proposals: List[Tuple[int, int]] = proposal_fn(relative_bounding_boxes)
    links: List[Tuple[int, int]] = list(
        filter(
            lambda indices: prediction_fn(
                (
                    relative_bounding_boxes[indices[0]],
                    relative_bounding_boxes[indices[1]],
                )
            )
            if indices[0] != indices[1]
            else False,
            proposals,
        )
    )
    G = nx.from_edgelist(links)
    connected_components: List[Set[int]] = nx.connected_components(G)  # type: ignore
    clusters: List[List[RelativeBoundingBox]] = [
        [relative_bounding_boxes[node] for node in subgraph]
        for subgraph in connected_components
    ]
    used_indices: Set[int] = set(
        [item for subset in nx.connected_components(G) for item in subset]
    )
    unused_indices: Set[int] = set(range(len(relative_bounding_boxes))) - used_indices

    for index in unused_indices:
        clusters.append([relative_bounding_boxes[index]])
    return [sorted(c, key=lambda ann: ann.center[0]) for c in clusters]
