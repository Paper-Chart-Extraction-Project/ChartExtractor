"""Performs clustering of labels based on their bounding boxes via various methods.

This module contains functions for clustering labels based on their bounding boxes. It additionally contains the function
"""

# Built-in imports
import re
from typing import List, Literal
from functools import reduce

# External imports
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Internal imports
from utilities.annotations import BoundingBox
from label_clustering.cluster import Cluster


def __cluster_kmeans(
    bounding_boxes: List[BoundingBox], possible_nclusters: List[int]
) -> List[int]:
    """
    Cluster bounding boxes using K-Means clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.608, mAP on number clusters: 0.879 with error.

    Args:
        `bounding_boxes` (List[BoundingBox]): 
            List of bounding boxes in YOLO format.
        `possible_nclusters` (List[int]): 
            List of possible number of clusters to try.

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


def __cluster_dbscan(
    bounding_boxes: List[BoundingBox], defined_eps: float, min_samples: int
) -> List[int]:
    """
    Cluster bounding boxes density based spatial clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.615, mAP on number clusters: 0.877 with error.

    Args:
        `bounding_boxes` (List[BoundingBox]): 
            List of bounding boxes.
        `defined_eps` (float): 
            Maximum distance between two samples to be in the neighborhood of one 
            another (center of BB).
        `min_samples` (int): 
            The number of samples (or total weight) for a point to be considered as core

    Returns:
        List of cluster labels.
    """
    # Convert to a NumPy array (using only x_center and y_center)
    data = np.array([box.center for box in bounding_boxes])

    # DBSCAN
    scan = DBSCAN(eps=defined_eps, min_samples=min_samples)
    labels = scan.fit_predict(data)

    return labels


def __cluster_agglomerative(
    bounding_boxes: List[BoundingBox], possible_nclusters: List[int]
) -> List[int]:
    """
    Cluster bounding boxes using agglomerative clustering algorithm.
    mAP on time clusters: 0.842, mAP on number clusters: 0.913 without error.
    mAP on time clusters: 0.588, mAP on number clusters: 0.875 with error.

    Args:
        `bounding_boxes` (List[BoundingBox]): 
            List of bounding boxes in YOLO format.
        `possible_nclusters` (List[int]): 
            List of possible number of clusters to try.

    Returns:
        List of cluster labels.
    """
    # make the bonding box data into a Numpy array
    data = np.array([box.center for box in bounding_boxes])

    # follow suit of the __cluster_kmeans algorithm to measure accuracy through silhoutte scores
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


def __time_correction(clusters: List[Cluster]) -> List[Cluster]:
    """
    Takes a list of time clusters and turns them from including repeats to not including repeats.
    There will be multiple 0, 5, 10, and so on. But the first (furthest to the left) is the true 
    one. The rest are repeats and will be changed to 60, 65, 70, and so on.

    Args:
        `clusters` (List[Cluster]): 
            List of Cluster objects.

    Returns:
        List of Cluster objects with corrected time labels.
    """
    # See if any repeats and identify them
    count_dict = dict()
    for cluster in clusters:
        label = cluster.label
        if label in count_dict:
            count_dict[label].append(cluster)
        else:
            count_dict[label] = [cluster]

    # Now iterate over the dictionary and find the labels with many bounding boxes. 
    # Lets change the labels of these.
    for label, clusters in count_dict.items():
        if len(clusters) > 1:
            # Sort by x
            sorted_clusters = sorted(
                clusters, key=lambda x: float(x.bounding_box.center[0])
            )
            # The one furthest to the left is the true one for the label.
            # For the rest add 60 to them depending on their index.
            for ix, cluster in enumerate(sorted_clusters):
                correct_label = (
                    f"{str(int(re.findall(r'\d+', label)[0]) + (ix * 60))}_mins"
                )
                sorted_clusters[ix] = Cluster(cluster.bounding_boxes, correct_label)

    clusters = reduce(lambda x, y: x + y, count_dict.values())

    return clusters


def cluster_boxes(
    bounding_boxes: List[BoundingBox],
    method: Literal["kmeans", "dbscan", "agglomerative"],
    unit: Literal["mins", "mmhg"],
    possible_nclusters: List[int] = None,
    defined_eps: float = None,
    min_samples: int = None,
) -> List[Cluster]:
    """
    Cluster bounding boxes using the specified method.

    Args:
        `bounding_boxes` (List[BoundingBox]): 
            List of BoundingBox objects to cluster based on location.
        `method` (Literal): 
            The clustering method to use. Can be "kmeans", "dbscan", or "agglomerative".
        `unit`: 
            The unit of the bounding boxes. Can be "mins" or "mmhg".
        `possible_nclusters` (List[int]): 
            A list of possible number of clusters to try. This is only used for KMeans and Agglomerative.
        `defined_eps` (float): 
            Maximum distance between two samples to be in the neighborhood of one another. 
            This is only used for DBSCAN.
        `min_samples` (int): 
            The number of samples (or total weight) for a point to be considered as core. 
            This is only used for DBSCAN.

    Returns:
        A list of Cluster objects based on the clustering results.
    """
    # See if the method is valid
    if method not in ["kmeans", "dbscan", "agglomerative"]:
        raise ValueError(f"Invalid method: {method}")
    # Ensure that the tuning parameters are passed for the method
    if method == "kmeans" and possible_nclusters is None:
        raise ValueError("possible_nclusters must be passed for KMeans.")
    if method == "dbscan" and (defined_eps is None or min_samples is None):
        raise ValueError("defined_eps and min_samples must be passed for DBSCAN.")
    if method == "agglomerative" and possible_nclusters is None:
        raise ValueError("possible_nclusters must be passed for Agglomerative.")
    # Ensure unit is valid
    if unit not in ["mins", "mmhg"]:
        raise ValueError(f"Invalid unit: {unit}")

    # Perform the clustering
    if method == "kmeans":
        labels = __cluster_kmeans(bounding_boxes, possible_nclusters)
    elif method == "dbscan":
        labels = __cluster_dbscan(bounding_boxes, defined_eps, min_samples)
    else:
        labels = __cluster_agglomerative(bounding_boxes, possible_nclusters)

    # Return a list Cluster objects based on the clustering results
    clusters = []
    for label in set(labels):
        cluster_bounding_boxes = [
            bounding_boxes[i] for i in range(len(labels)) if labels[i] == label
        ]
        clusters.append(Cluster.from_boxes_and_unit(cluster_bounding_boxes, unit))

    if unit == "mins":
        clusters = __time_correction(clusters)

    return clusters
