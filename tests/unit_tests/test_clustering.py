"""Tests clustering modules"""
# NOTE: this relies on the test data in the data folder, which is not included in the repository.
# If you would like to run these tests, please reach out to the author for the data.
# The main purpose is for demonstrating how to use the clustering methods.

# Built-in Libraries
import os
import sys
import json

# External Libraries
import unittest

# Internal Libraries
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
from label_clustering.cluster import Cluster
from label_clustering.clustering_methods import cluster_boxes
from label_clustering.isolate_labels import extract_relevant_bounding_boxes


class TestClustering(unittest.TestCase):
    """Tests the clustering methods."""

    # Load yolo_data.json
    with open(os.path.join("test_data", "yolo_data.json")) as json_file:
        test_data = list(json.load(json_file).values())[0]

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
        # There should be 62 labels in total
        assert len(time_clusters + number_clusters) == 62

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
        # There should be 62 labels in total
        assert len(time_clusters + number_clusters) == 62

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
        # There should be 62 labels in total
        assert len(time_clusters + number_clusters) == 62


if __name__ == "__main__":
    unittest.main(verbosity=4)
