"""A module that reads and writes yolo datasets to the intermediate VisionDataset representation."""

import warnings
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, NotRequired, Required, TypedDict

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from .exceptions import YoloDatasetError
from .warnings import YoloDatasetWarning


class _YoloDatasetType(StrEnum):
    """An enum representing the different types of YOLO datasets."""

    DETECT = "detect"
    POSE = "pose"


type _RawYamlData = dict[str, Any]


class _YoloDatasetPaths(TypedDict):
    """A dictionary containing the paths to the dataset.

    Used internally by _YoloDatasetConfig. If a path is needed from the config, it should go
    through the methods provided by _YoloDatasetConfig, which will attempt to resolve the paths
    relative to the base path.

    Requires there to a be a training images path and validation images path.
    Optionally, a test images path, or a base path can exist.
    """

    base: NotRequired[Path]
    train_images: Required[Path]
    val_images: Required[Path]
    test_images: NotRequired[Path]


class _YoloDatasetConfig(BaseModel):
    """Reads, validates, and stores the data of the yolo dataset.yaml config file.

    Attributes:
        dataset_type (_YoloDatasetType):
            The type of the yolo dataset (detect or pose).
        training_images_path (Path):
            The folder path to the training images.
        val_images_path (Path):
            The folder path to the validation images.
        test_images_path (Path | None):
            The folder path to the testing images.
        training_labels_path (Path):
            The folder path to the training labels.
        val_labels_path (Path):
            The folder path to the validation labels.
        test_labels_path (Path | None):
            The folder path to the testing labels.
        evaluation_set_image_paths (dict[str, Path]):
            A dictionary mapping the name of evaluation sets to the folderpath that contains
            their images.
        evaluation_set_label_paths (dict[str, Path]):
            A dictionary mapping the name of evaluation sets to the folderpath that contains
            their labels.
    """

    model_config = ConfigDict(frozen=True)

    raw_yaml_data: _RawYamlData

    @field_validator("raw_yaml_data", mode="before")
    @classmethod
    def necessary_keys_exist_in_raw_yaml_data(
        cls, raw_yaml_data: _RawYamlData
    ) -> _RawYamlData:
        """Checks if the necessary keys exist in the raw yaml dictionary."""
        necessary_keys: set[str] = {"train", "val", "names"}
        overlapping_keys: set[str] = set(raw_yaml_data.keys()).intersection(
            necessary_keys
        )
        all_necessary_keys_exist: bool = overlapping_keys == necessary_keys

        if not all_necessary_keys_exist:
            keys_that_dont_exist: set[str] = necessary_keys - overlapping_keys
            err_msg: str = (
                "Not all necessary keys exist in the dataset yaml file.\nKeys that don't exist "
                + f"in the yaml file: {', '.join(sorted(keys_that_dont_exist))}."
            )

            raise YoloDatasetError(err_msg)

        return raw_yaml_data

    @model_validator(mode="after")
    def all_path_parent_dirs_is_image(self) -> _YoloDatasetConfig:
        """Ensures that the parent directory of all the image paths is 'images'."""
        for split in ["train_images", "val_images", "test_images"]:
            if self._raw_paths.get(split) is None:
                continue
            full_folderpath: Path = self._raw_paths[split]
            if full_folderpath.parent.name != "images":
                raise YoloDatasetError(
                    f"The parent directory of split path {full_folderpath} is not 'images'."
                )

        return self

    @model_validator(mode="after")
    def warn_if_relative_path_used_for_base_path(self) -> _YoloDatasetConfig:
        """Warns the user if they supply a yaml file where the base path is relative."""
        base_path: Path | None = self._raw_paths.get("base")
        if base_path is None:
            return self

        if base_path.is_absolute():
            return self

        warnings.warn(
            "Base path is a relative path, which may lead to issues.",
            YoloDatasetWarning,
        )
        return self

    @cached_property
    def _raw_paths(self) -> _YoloDatasetPaths:
        """A typed dictionary of paths to the dataset's image folders."""

        yolo_dataset_paths: _YoloDatasetPaths = {}

        def add_raw_path_to_dict(
            value_to_get: str, rename_to: str
        ) -> _YoloDatasetPaths:
            if value_to_get not in self.raw_yaml_data:
                return yolo_dataset_paths

            raw_value: Any = self.raw_yaml_data[value_to_get]

            if isinstance(raw_value, str):
                yolo_dataset_paths[rename_to] = Path(raw_value)
            else:
                raise ValidationError(
                    f"The value for '{value_to_get}' in the yaml data was expected to be a "
                    + f"string, but was {type(raw_value)}."
                )

            return yolo_dataset_paths

        pathes: dict[str, str] = {
            "path": "base",
            "train": "train_images",
            "val": "val_images",
            "test": "test_images",
        }
        for yaml_key, dataset_path_key in pathes.items():
            yolo_dataset_paths: _YoloDatasetPaths = add_raw_path_to_dict(
                yaml_key, dataset_path_key
            )

        return yolo_dataset_paths

    @cached_property
    def dataset_type(self) -> _YoloDatasetType:
        """Determines and returns the type of the yolo dataset."""
        keys_in_raw_yaml_data: set[str] = set(self.raw_yaml_data.keys())
        pose_specific_keys: set[str] = {"kpt_shape", "kpt_names"}
        pose_specific_keys_in_raw_yaml_data: set[str] = (
            pose_specific_keys & keys_in_raw_yaml_data
        )

        match sorted(pose_specific_keys_in_raw_yaml_data):
            case ["kpt_names", "kpt_shape"]:
                return _YoloDatasetType.POSE
            case []:
                return _YoloDatasetType.DETECT
            case ["kpt_names"]:
                raise KeyError(
                    "Yaml data contained a value for 'kpt_names' but not 'kpt_shape'."
                )
            case ["kpt_shape"]:
                raise KeyError(
                    "Yaml data contained a value for 'kpt_shape' but not 'kpt_names'."
                )

    @cached_property
    def train_images_path(self) -> Path:
        """The path to the training images directory."""
        train_image_path: Path = self._raw_paths["train_images"]

        if "base" in self._raw_paths:
            base_path: Path = self._raw_paths["base"]
            return base_path / train_image_path

        return train_image_path

    @cached_property
    def val_images_path(self) -> Path:
        """The full path to the validation images directory."""
        val_image_path: Path = self._raw_paths["val_images"]

        if "base" in self._raw_paths:
            base_path: Path = self._raw_paths["base"]
            return base_path / val_image_path

        return val_image_path

    @cached_property
    def test_images_path(self) -> Path | None:
        """The full path to the test images directory, if it exists."""
        if "test_path" not in self._raw_paths:
            return None

        test_image_path: Path = self._raw_paths["test_images"]

        if "base" in self._raw_paths:
            base_path: Path = self._raw_paths["base"]
            return base_path / test_image_path

        return test_image_path

    @cached_property
    def train_labels_path(self) -> Path:
        """The full path to the training labels directory."""
        train_folder_name: str = self.train_images_path.name

        return self.train_images_path.parent.parent / "labels" / train_folder_name

    @cached_property
    def val_labels_path(self) -> Path:
        """The full path to the validation labels directory."""
        val_folder_name: str = self.val_images_path.name

        return self.val_images_path.parent.parent / "labels" / val_folder_name

    @cached_property
    def test_labels_path(self) -> Path | None:
        """The full path to the test labels directory, if it exists."""
        if self.test_images_path is None:
            return None

        test_folder_name: str = self.test_images_path.name

        return self.test_images_path.parent.parent / "labels" / test_folder_name

    @classmethod
    def from_path(cls, path_to_dataset_yaml: Path) -> _YoloDatasetConfig:
        """Creates a _YoloDatasetConfig object by reading data from a yaml file.

        Args:
            path_to_dataset_yaml (Path):
                The filepath to the dataset yaml file.

        Returns:
            A _YoloDatasetConfig object storing validated data from the dataset yaml file.
        """
        raise NotImplementedError()
