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
from label_clustering.cluster import Cluster  # Needs to be tested.
from label_clustering.clustering_methods import cluster_boxes
from label_clustering.isolate_labels import extract_relevant_bounding_boxes


class TestClustering(unittest.TestCase):
    """Tests the clustering methods."""

    # Load yolo_data.json
    with open(os.path.join("test_data", "yolo_data.json")) as json_file:
        test_data = list(json.load(json_file).values())[0]

    expected_time_values = [
        "0_mins",
        "5_mins",
        "10_mins",
        "15_mins",
        "20_mins",
        "25_mins",
        "30_mins",
        "35_mins",
        "40_mins",
        "45_mins",
        "50_mins",
        "55_mins",
        "60_mins",
        "65_mins",
        "70_mins",
        "75_mins",
        "80_mins",
        "85_mins",
        "90_mins",
        "95_mins",
        "100_mins",
        "105_mins",
        "110_mins",
        "115_mins",
        "120_mins",
        "125_mins",
        "130_mins",
        "135_mins",
        "140_mins",
        "145_mins",
        "150_mins",
        "155_mins",
        "160_mins",
        "165_mins",
        "170_mins",
        "175_mins",
        "180_mins",
        "185_mins",
        "190_mins",
        "195_mins",
        "200_mins",
        "205_mins",
    ]

    expected_number_values = [
        "30_mmhg",
        "40_mmhg",
        "50_mmhg",
        "60_mmhg",
        "70_mmhg",
        "80_mmhg",
        "90_mmhg",
        "100_mmhg",
        "110_mmhg",
        "120_mmhg",
        "130_mmhg",
        "140_mmhg",
        "150_mmhg",
        "160_mmhg",
        "170_mmhg",
        "180_mmhg",
        "190_mmhg",
        "200_mmhg",
        "210_mmhg",
        "220_mmhg",
    ]

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
