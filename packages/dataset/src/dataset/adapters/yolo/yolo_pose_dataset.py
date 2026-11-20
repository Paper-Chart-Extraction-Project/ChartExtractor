"""A module that can read/write yolo pose datasets to PoseVisionDataset."""

from collections import namedtuple
from functools import cached_property
from typing import Self

from pydantic import model_validator

from .exceptions import YoloDatasetConfigError
from .yolo_dataset import _YoloDatasetConfig

KeypointShape = namedtuple("KeypointShape", ["num_keypoints", "num_dimensions"])


class _YoloPoseDatasetConfig(_YoloDatasetConfig):
    """Reads, validates, and stores data from a yolo pose dataset.yaml file.

    There are no differences between the base class and this one; it exists for typing purposes.
    """

    @model_validator(mode="after")
    def all_keypoint_keys_exist(self) -> Self:
        """Ensures all the keypoint keys exist if the config is for a pose dataset."""
        keys_in_raw_yaml_data: set[str] = set(self.raw_yaml_data.keys())
        pose_specific_keys: set[str] = {"kpt_shape", "kpt_names"}
        pose_specific_keys_in_raw_yaml_data: set[str] = (
            pose_specific_keys & keys_in_raw_yaml_data
        )
        if pose_specific_keys_in_raw_yaml_data != pose_specific_keys:
            missing_keys: str = ",".join(
                f"'{mk}'"
                for mk in pose_specific_keys - pose_specific_keys_in_raw_yaml_data
            )
            raise YoloDatasetConfigError(
                f"Error parsing pose dataset yaml file.\nCannot find the key(s) {missing_keys}."
            )

        return self

    @model_validator(mode="after")
    def validate_kpt_shape(self) -> Self:
        """Calls kpt_shape just to run its own validation."""
        if len(self.raw_yaml_data["kpt_shape"]) != 2:
            raise YoloDatasetConfigError(
                "The shape of the 'kpt_shape' parameter in the yolo config is not 2. "
                f"Expected [num_keypoints, num_dimensions], got {self.raw_yaml_data['kpt_shape']}."
            )

        num_keypoints: int = self.raw_yaml_data["kpt_shape"][0]
        num_dimensions: int = self.raw_yaml_data["kpt_shape"][1]

        if not isinstance(num_keypoints, int):
            raise YoloDatasetConfigError(
                f"Expected number of keypoints to be an integer, got {type(num_keypoints)!s}."
            )
        if not isinstance(num_dimensions, int):
            raise YoloDatasetConfigError(
                "Expected number of keypoint dimensions to be an integer, got "
                f"{type(num_dimensions)!s}."
            )
        if num_dimensions not in [2, 3]:
            raise YoloDatasetConfigError(
                f"Expected number of keypoint dimensions is 2 or 3, got {num_dimensions}."
            )

        if num_keypoints != len(self.kpt_names):
            raise YoloDatasetConfigError(
                f"Number of keypoints in kpt_shape ({num_keypoints}) does not equal the number "
                f"of keypoints indicated by kpt_names ({len(self.kpt_names)})."
            )
        return self

    @cached_property
    def kpt_names(self) -> list[list[str]]:
        """A list of the names of keypoints."""
        return [
            self.raw_yaml_data["kpt_names"][sk]
            for sk in sorted(self.raw_yaml_data["kpt_names"].keys())
        ]

    @cached_property
    def kpt_shape(self) -> KeypointShape:
        """A tuple containing the number of keypoints and 'dimensions' of each keypoint."""
        num_keypoints: int = self.raw_yaml_data["kpt_shape"][0]
        num_dimensions: int = self.raw_yaml_data["kpt_shape"][1]

        return KeypointShape(num_keypoints=num_keypoints, num_dimensions=num_dimensions)
