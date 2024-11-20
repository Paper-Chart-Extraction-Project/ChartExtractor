"""Module containing the cluster class

This module contains the class for clustered bounding boxes.
Once sorted into clusters, these bounding boxes are used to initialize these class, which will automatically give them a label.
It additionally creates a new bounding box that encompasses all the bounding boxes in the cluster.
"""

# Built-in imports
from functools import cached_property
from typing import List, Literal

# External imports
# ...

# Internal imports
from utilities.annotations import BoundingBox


class Cluster:
    """
    Class for clustered bounding boxes.
    """

    def __init__(
        self, bounding_boxes: List[BoundingBox], expected_unit: Literal["mmhg", "mins"]
    ) -> None:
        """
        Initialize the Cluster class with a list of bounding boxes.

        Args:
            bounding_boxes: List of bounding boxes in YOLO format.
            expected_unit: Expected unit of the bounding boxes.

        Returns:
            None
        """
        self.bounding_boxes = bounding_boxes
        self.label = self.__create_cluster_label(expected_unit)

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """
        Create a bounding box that encompasses all the bounding boxes in the cluster.

        Args:
            None

        Returns:
            A bounding box in YOLO format.
        """
        left = min([bb.left for bb in self.bounding_boxes])
        right = max([bb.right for bb in self.bounding_boxes])
        top = min([bb.top for bb in self.bounding_boxes])
        bottom = max([bb.bottom for bb in self.bounding_boxes])
        return BoundingBox(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            category=self.label,
        )

    def update_label(self, new_label: str) -> None:
        """
        Update the label of the cluster.
        This should be used in these situiation:
            - If the ground truth label is determined to be different from the given label based on the spacing between clusters.
            - To update the time labels to increase by 60 seconds.

        Args:
            new_label: The new label of the cluster.

        Returns:
            None
        """
        self.label = new_label

    def __create_cluster_label(
        self,
        unit: Literal["mmhg", "mins"],
    ) -> str:
        """
        Create a label for the cluster based on the bounding boxes.

        Args:
            bounding_boxes: List of bounding boxes in as instances of the BoundingBox class.
            unit: The unit of the bounding boxes. Can be either "mmhg" or "mins".

        Returns:
            The label of the cluster as a string.
        """
        sorted_bbs = sorted(self.bounding_boxes, key=lambda x: float(x.left))
        categories = [element.category for element in sorted_bbs]
        # Turn list of strings into a string
        label = f"{''.join(categories)}_{unit}"
        return label
