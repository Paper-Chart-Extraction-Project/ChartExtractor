"""Tests clustering modules"""

# NOTE: this relies on the test data in the data folder, which is not included in the repository.
# If you would like to run these tests, please reach out to the author for the data.
# The main purpose is for demonstrating how to use the clustering methods.

# Built-in Libraries
import os
import sys
from typing import List
import json

# External Libraries
import unittest

# Internal Libraries
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
from label_clustering.cluster import Cluster  # Needs to be tested.
from label_clustering.clustering_methods import cluster_boxes
from label_clustering.isolate_labels import extract_relevant_bounding_boxes


class TestClustering(unittest.TestCase):
    """Tests the clustering methods."""

    # Load yolo_data.json
    with open(os.path.join("test_data", "yolo_data.json")) as json_file:
        test_data = list(json.load(json_file).values())[0]

    # 0_mins, 5_mins, 10_mins, ..., 200_mins, 205_mins.
    expected_time_values: List[str] = [f"{t*5}_mins" for t in range(0, 210 // 5)]
    # 30_mmhg, 40_mmhg, 50_mmhg, ..., 210_mmhg, 220_mmhg.
    expected_number_values: List[str] = [f"{m*10}_mmhg" for m in range(3, 230 // 10)]

    def test_1_extract_relevant_bounding_boxes(self):
        """Tests the extract_relevant_bounding_boxes function."""
        bounding_boxes = extract_relevant_bounding_boxes(self.test_data)
        assert len(bounding_boxes[0]) == 76 and len(bounding_boxes[1]) == 53

    def test_2_cluster_kmeans(self):
        """Tests the cluster_kmeans function."""
        bounding_boxes = extract_relevant_bounding_boxes(self.test_data)
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method="kmeans",
            possible_nclusters=[40, 41, 42],
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method="kmeans",
            possible_nclusters=[18, 19, 20],
            unit="mmhg",
        )

        assert (
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(self.expected_time_values)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(self.expected_number_values)
        )

    def test_3_cluster_dbscan(self):
        """Tests the cluster_dbscan function."""
        bounding_boxes = extract_relevant_bounding_boxes(self.test_data)
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method="dbscan",
            defined_eps=5,
            min_samples=1,
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method="dbscan",
            defined_eps=5,
            min_samples=2,
            unit="mmhg",
        )

        assert (
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(self.expected_time_values)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(self.expected_number_values)
        )

    def test_4_cluster_agglomerative(self):
        """Tests the cluster_agglomerative function."""
        bounding_boxes = extract_relevant_bounding_boxes(self.test_data)
        time_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[0],
            method="agglomerative",
            possible_nclusters=[40, 41, 42],
            unit="mins",
        )
        number_clusters = cluster_boxes(
            bounding_boxes=bounding_boxes[1],
            method="agglomerative",
            possible_nclusters=[18, 19, 20],
            unit="mmhg",
        )

        assert (
            len(time_clusters + number_clusters) == 62
            and len(time_clusters) == 42
            and len(number_clusters) == 20
            and set([cluster.get_label() for cluster in time_clusters])
            == set(self.expected_time_values)
            and set([cluster.get_label() for cluster in number_clusters])
            == set(self.expected_number_values)
        )


if __name__ == "__main__":
    unittest.main(verbosity=4)
