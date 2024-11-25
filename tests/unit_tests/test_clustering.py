"""Tests clustering modules"""

# NOTE: this relies on the test data in the data folder, which is not included in the repository.
# If you would like to run these tests, please reach out to the author for the data.
# The main purpose is for demonstrating how to use the clustering methods.

# Built-in Libraries
from functools import partial
import os
import sys
from typing import Callable, Dict, List
import json

# External Libraries
import pytest

# Internal Libraries
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
from label_clustering.cluster import Cluster  # Needs to be tested.
from label_clustering.clustering_methods import (
    cluster_boxes,
    __cluster_agglomerative,
    __cluster_dbscan,
    __cluster_kmeans,
)
from label_clustering.isolate_labels import extract_relevant_bounding_boxes
from utilities.annotations import BoundingBox


# 0_mins, 5_mins, 10_mins, ..., 200_mins, 205_mins.
EXPECTED_TIME_VALUES: List[str] = [f"{t*5}_mins" for t in range(0, 210 // 5)]
# 30_mmhg, 40_mmhg, 50_mmhg, ..., 210_mmhg, 220_mmhg.
EXPECTED_NUMBER_VALUES: List[str] = [f"{m*10}_mmhg" for m in range(3, 230 // 10)]


@pytest.fixture
def test_data() -> Dict:
    """Test data for TestClustering."""
    with open(os.path.join("test_data", "yolo_data.json")) as json_file:
        data: Dict = list(json.load(json_file).values())[0]
    return data


class TestClustering:
    """Tests the clustering methods."""

    def test_extract_relevant_bounding_boxes(self, test_data):
        """Tests the extract_relevant_bounding_boxes function."""
        bounding_boxes = extract_relevant_bounding_boxes(test_data)
        assert len(bounding_boxes[0]) == 76 and len(bounding_boxes[1]) == 53

    def test_cluster_kmeans(self, test_data):
        """Tests the cluster_kmeans function."""
        bounding_boxes: List[BoundingBox] = extract_relevant_bounding_boxes(test_data)
        time_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_kmeans, possible_nclusters=[40, 41, 42]
        )
        number_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_kmeans, possible_nclusters=[18, 19, 20]
        )
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method=time_clustering_func,
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method=number_clustering_func
            unit="mmhg",
        )

        assert(
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(EXPECTED_TIME_VALUES)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(EXPECTED_NUMBER_VALUES)
        )

    def test_cluster_dbscan(self, test_data):
        """Tests the cluster_dbscan function."""
        bounding_boxes = extract_relevant_bounding_boxes(test_data)
        time_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_dbscan, defined_eps=5, min_samples=1
        )
        number_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_dbscan, defined_eps=5, min_samples=2
        )
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method=time_clustering_func,
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method=number_clustering_func,
            unit="mmhg",
        )

        assert(
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(EXPECTED_TIME_VALUES)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(EXPECTED_NUMBER_VALUES)
        )

    def test_cluster_agglomerative(self, test_data):
        """Tests the cluster_agglomerative function."""
        bounding_boxes = extract_relevant_bounding_boxes(test_data)
        time_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_agglomerative, possible_nclusters=[40, 41, 42]
        )
        number_clustering_func: Callable[[List[BoundingBox]], List[Cluster]] = partial(
            __cluster_agglomerative, possible_nclusters=[18, 19, 20]
        )
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method=time_clustering_func,
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method=number_clustering_func
            unit="mmhg",
        )

        assert(
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(EXPECTED_TIME_VALUES)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(EXPECTED_NUMBER_VALUES)
        )
