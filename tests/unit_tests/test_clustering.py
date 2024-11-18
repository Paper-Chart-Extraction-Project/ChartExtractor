"""Tests clustering modules"""

# Built-in Libraries
import os
import sys

# External Libraries
import pytest

# Internal Libraries
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
from label_clustering.cluster import Cluster
from label_clustering.clustering_methods import (
    cluster_kmeans,
    cluster_dbscan,
    cluster_agglomerative,
)
from label_clustering.isolate_labels import extract_relevant_bounding_boxes
