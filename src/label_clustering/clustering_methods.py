"""Performs clustering of labels based on their bounding boxes via various methods.

This module contains functions for clustering labels based on their bounding boxes. It additionally contains the function
"""

# Built-in imports
from typing import List, Dict

# External imports
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Internal imports
from ..utilities.annotations import BoundingBox


def cluster_kmeans(
    bounding_boxes: List[BoundingBox], possible_nclusters: List[int]
) -> List[int]:
    """
    Cluster bounding boxes using K-Means clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.608, mAP on number clusters: 0.879 with error.

    Args:
        bounding_boxes: List of bounding boxes in YOLO format.
        possible_nclusters: List of possible number of clusters to try.

    Returns:
        List of cluster labels.
    """
    # Convert to a NumPy array (using only x_center and y_center)
    data = np.array([box.center for box in bounding_boxes])

    cluster_performance_map = {}
    for number_of_clusters in possible_nclusters:
        if number_of_clusters > len(data):
            raise (
                f"Number of clusters {number_of_clusters} is greater than number of bounding boxes {len(data)}."
            )
        if number_of_clusters < 1:
            raise (f"Number of clusters {number_of_clusters} must be greater than 0.")
        # Apply K-Means
        kmeans = KMeans(
            n_clusters=number_of_clusters,
            init="k-means++",
            n_init=20,
            max_iter=500,
            tol=1e-8,
            random_state=42,
        )
        kmeans.fit(data)

        # Get cluster labels
        labels = kmeans.predict(data)
        silhouette_avg = silhouette_score(data, labels)

        # print(
        #     f"Number of clusters: {number_of_clusters}, Silhouette score: {silhouette_avg}"
        # )

        cluster_performance_map[number_of_clusters] = {
            "score": silhouette_avg,
            "labels": labels,
        }

    # Evaluate the performance of each number of clusters and select the one with the highest silhouette score
    # if it is 0.003 greater than what should be the number of clusters otherwise go with proper_nclusters
    n_clusters_max_silhouette = max(
        cluster_performance_map, key=lambda x: cluster_performance_map[x]["score"]
    )
    best_n_clusters = (
        n_clusters_max_silhouette
        if (
            (
                cluster_performance_map[n_clusters_max_silhouette]["score"]
                - cluster_performance_map[max(possible_nclusters)]["score"]
            )
            >= 0.005
        )
        else max(possible_nclusters)
    )
    return cluster_performance_map[best_n_clusters]["labels"]


def cluster_dbscan(
    bounding_boxes: List[BoundingBox], defined_eps: float, min_samples: int
) -> List[int]:
    """
    Cluster bounding boxes density based spatial clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.615, mAP on number clusters: 0.877 with error.

    Args:
        bounding_boxes: List of bounding boxes.
        defined_eps: Maximum distance between two samples to be in the neighborhood of one another (center of BB).
        min_samples: The number of samples (or total weight) for a point to be considered as core

    Returns:
        List of cluster labels.
    """
    # Convert to a NumPy array (using only x_center and y_center)
    data = np.array([box.center for box in bounding_boxes])

    # DBSCAN
    scan = DBSCAN(eps=defined_eps, min_samples=min_samples)
    labels = scan.fit_predict(data)

    return labels


def cluster_agglomerative(
    bounding_boxes: List[BoundingBox], possible_nclusters: List[int]
) -> List[int]:
    """
    Cluster bounding boxes using agglomerative clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.588, mAP on number clusters: 0.875 with error.

    Args:
        bounding_boxes: List of bounding boxes in YOLO format.
        possible_nclusters: List of possible number of clusters to try.

    Returns:
        List of cluster labels.
    """
    # make the bonding box data into a Numpy array
    data = np.array([box.center for box in bounding_boxes])

    # follow suit of the cluster_kmeans algorithm to measure accuracy through silhoutte scores
    cluster_performance_map = {}
    for number_of_clusters in possible_nclusters:
        if number_of_clusters > len(data):
            raise (
                f"Number of clusters {number_of_clusters} is greater than number of bounding boxes {len(data)}."
            )
        if number_of_clusters < 1:
            raise (f"Number of clusters {number_of_clusters} must be greater than 0.")
        # use agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=number_of_clusters, linkage="single")
        # get labels
        labels = agg.fit_predict(data)
        # compute the silhoutte scores
        silhouette_avg = silhouette_score(data, labels)

        cluster_performance_map[number_of_clusters] = {
            "score": silhouette_avg,
            "labels": labels,
        }

    # get the number of clusters with the best silhoutte score
    n_clusters_max_silhouette = max(
        cluster_performance_map, key=lambda x: cluster_performance_map[x]["score"]
    )

    best_n_clusters = (
        n_clusters_max_silhouette
        if (
            (
                cluster_performance_map[n_clusters_max_silhouette]["score"]
                - cluster_performance_map[max(possible_nclusters)]["score"]
            )
            >= 0.003
        )
        else max(possible_nclusters)
    )
    return cluster_performance_map[best_n_clusters]["labels"]


def cluster_to_bounding_boxes(
    labels: List[str], bounding_boxes: List[BoundingBox]
) -> Dict[str, BoundingBox]:
    """
    Create a dictionary with cluster labels as keys and a BoundingBox for the cluster.

    Args:
        labels: List of cluster labels.
        bounding_boxes: List of bounding boxes.

    Returns:
        Dictionary with cluster labels as keys and a bounding box value as values.
    """
    # Create a dictionary to store labelled elements
    label_dict = {}

    # Iterate over both lists for labels and the bounding boxes found
    for label, box in zip(labels, bounding_boxes):
        label = int(label)
        if label not in label_dict:
            # Create a new list for this label if it doesn't exist
            label_dict[label] = []
        # Append the element to the corresponding label list
        label_dict[label].append(box)

    # Create dictionary that will hold the cluster label and bounding box
    cluster_dict = {}
    # Iterate over the label_dict to find the overall coordinates for the cluster
    for key in label_dict:
        # calculate the coordinates of the cluster bounding box
        x_left = min([bb.left for bb in label_dict[key]])
        x_right = max([bb.right for bb in label_dict[key]])
        y_top = min([bb.top for bb in label_dict[key]])
        y_bottom = max([bb.bottom for bb in label_dict[key]])
        # get the category based off of digit detections
        sorted_boxes = sorted(label_dict[key], key=lambda x: float(x.left))
        sorted_categories = [bb.category for bb in sorted_boxes]
        # Turn list of strings into a string
        cluster_category = f"{''.join(sorted_categories)}"
        # store the bounding box into the dictionary
        cluster_dict[key] = BoundingBox(
            category=cluster_category,
            left=x_left,
            right=x_right,
            top=y_top,
            bottom=y_bottom,
        )
    return cluster_dict