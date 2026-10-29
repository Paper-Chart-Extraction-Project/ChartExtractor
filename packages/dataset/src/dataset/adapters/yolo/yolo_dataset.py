"""A module that reads and writes yolo datasets to the intermediate VisionDataset representation."""

from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, NotRequired, Required, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from .exceptions import YoloDatasetError


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

    _raw_yaml_data: _RawYamlData

    @field_validator("_raw_yaml_data", mode="before")
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
            err_msg: str = "Not all necessary keys exist in the dataset yaml file.\n"
            err_msg += f"Keys that don't exist in the yaml file: {keys_that_dont_exist}"

            raise YoloDatasetError(err_msg)

    @field_validator("_raw_paths")
    @classmethod
    def all_path_parent_dirs_is_image(
        cls, raw_paths: _YoloDatasetPaths
    ) -> _YoloDatasetPaths:
        """Ensures that the parent directory of all the image paths is 'images'."""
        for split in ["train", "val", "test"]:
            if raw_paths.get(split) is None:
                continue
            full_folderpath: Path = raw_paths[split]
            if full_folderpath.parent.name != "images":
                raise YoloDatasetError(
                    f"The parent directory of split path {full_folderpath} is not 'images'."
                )

        return raw_paths

    @cached_property
    def _raw_paths(self) -> _YoloDatasetPaths:
        """A typed dictionary of paths to the dataset's image folders."""
        raise NotImplementedError()

    @cached_property
    def dataset_type(self) -> _YoloDatasetType:
        """Determines and returns the type of the yolo dataset."""
        raise NotImplementedError()

    @cached_property
    def train_images_path(self) -> Path:
        """The path to the training images directory."""
        train_image_path: Path = self._raw_paths["train_images"]

        if "base_path" in self._raw_paths:
            base_path: Path = self._raw_paths["base_path"]
            return base_path / train_image_path

        return train_image_path

    @cached_property
    def val_images_path(self) -> Path:
        """The full path to the validation images directory."""
        val_image_path: Path = self._raw_paths["val_images"]

        if "base_path" in self._raw_paths:
            base_path: Path = self._raw_paths["base_path"]
            return base_path / val_image_path

        return val_image_path

    @cached_property
    def test_images_path(self) -> Path | None:
        """The full path to the test images directory, if it exists."""
        if "test_path" not in self._raw_paths:
            return None

        test_image_path: Path = self._raw_paths["test_images"]

        if "base_path" in self._raw_paths:
            base_path: Path = self._raw_paths["base_path"]
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
