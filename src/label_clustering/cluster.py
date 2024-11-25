"""Module containing the cluster class

This module contains the class for clustered bounding boxes.
Once sorted into clusters, these bounding boxes are used to initialize these class, which will 
automatically give them a label. It additionally creates a new bounding box that encompasses all 
the bounding boxes in the cluster.
"""

# Built-in imports
from functools import cached_property
from typing import List, Literal

# External imports
# ...

# Internal imports
from utilities.annotations import BoundingBox


class Cluster:
    """Class for clustered bounding boxes."""

    def __init__(self, bounding_boxes: List[BoundingBox], label: str) -> "Cluster":
        """Initialize the Cluster directly from its arguments.

        Args:
            `bounding_boxes` (List[BoundingBox]):
                List of bounding boxes in YOLO format.
            `label` (str):
                The exact label to use.
        """
        self.bounding_boxes = bounding_boxes
        self.label = label

    def __repr__(self) -> str:
        """A string representation of this object."""
        return f"Cluster(bounding_boxes={self.bounding_boxes}, label={self.label})"

    @classmethod
    def from_boxes_and_unit(
        cls, bounding_boxes: List[BoundingBox], expected_unit: Literal["mmhg", "mins"]
    ) -> "Cluster":
        """Initialize the Cluster class with a list of bounding boxes.

        Args:
            `bounding_boxes` (List[BoundingBox]):
                List of bounding boxes in YOLO format.
            `expected_unit` (Literal["mmhg", "mins"]):
                Expected unit of the bounding boxes.

        Returns:
            A new Cluster object.
        """
        label = cls.__create_cluster_label(bounding_boxes, expected_unit)
        return cls(bounding_boxes, label)

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """Create a bounding box that encompasses all the bounding boxes in the cluster.

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

    @staticmethod
    def __create_cluster_label(
        bounding_boxes: List[BoundingBox], unit: Literal["mmhg", "mins"]
    ) -> str:
        """
        Create a label for the cluster based on the bounding boxes.

        Args:
            `bounding_boxes` (List[BoundingBox]):
                List of bounding boxes in YOLO format.
            `unit` (Literal["mmhg", "mins"]):
                The unit of the bounding boxes. Can be either "mmhg" or "mins".

        Returns:
            The label of the cluster as a string.
        """
        sorted_bbs = sorted(bounding_boxes, key=lambda x: float(x.left))
        categories = [element.category for element in sorted_bbs]
        label = f"{''.join(categories)}_{unit}"
        return label
