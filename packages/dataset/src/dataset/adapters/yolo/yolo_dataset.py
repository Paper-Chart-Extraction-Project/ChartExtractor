"""A module that reads and writes yolo datasets to the intermediate VisionDataset representation."""

import os
import warnings
from collections import namedtuple
from collections.abc import Callable
from enum import StrEnum
from functools import cached_property
from operator import attrgetter
from pathlib import Path
from typing import Any, NotRequired, Required, TypedDict

import yaml
from core.annotations import Annotation, BoundingBox, BoundingBoxWithKeypoints
from core.geometry import Point, Rectangle
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from dataset import VisionDataset

from .exceptions import YoloDatasetError
from .warnings import YoloDatasetWarning


class _YoloDatasetType(StrEnum):
    """An enum representing the different types of YOLO datasets."""

    DETECT = "detect"
    POSE = "pose"


type _RawYamlData = dict[str, Any]


KeypointShape = namedtuple("KeypointShape", ["num_keypoints", "num_dimensions"])


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
        names (list[str]):
            The names of the bounding boxes in the dataset.
            Each annotation will have an index as the first entry that maps to an entry in this
            list.
        kpt_names (list[str]):
            The names of the keypoints in the dataset.
            Each keypoint annotation will have ordered keypoints. The position in that ordering
            will map to the name of the keypoint in this list.
        kpt_shape (KeypointShape):
            A namedtuple containing the (0) number of keypoints for each annotation and (1) the
            'dimensions' of each keypoint. The dimension can be either 2 or 3 where the first
            dimension is the x location of the keypoint, the second dimension is the y location
            of the keypoint, and the optional third is the 'visibility' of the keypoint. The
            visisbility can be either 0='Missing/Out of view', 1='Occluded', or 2='Visible'.
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
    def all_keypoint_keys_exist(self) -> "_YoloDatasetConfig":
        """Ensures all the keypoint keys exist if the config is for a pose dataset."""
        try:
            _ = self.dataset_type
        except KeyError as ke:
            raise YoloDatasetError(f"Error parsing pose dataset yaml file.\n{ke}")

        return self

    @model_validator(mode="after")
    def all_path_parent_dirs_is_image(self) -> "_YoloDatasetConfig":
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
    def warn_if_relative_path_used_for_base_path(self) -> "_YoloDatasetConfig":
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

        paths: dict[str, str] = {
            "path": "base",
            "train": "train_images",
            "val": "val_images",
            "test": "test_images",
        }
        for yaml_key, dataset_path_key in paths.items():
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
        if "test_images" not in self._raw_paths:
            return None

        test_image_path: Path = self._raw_paths["test_images"]

        if "base" in self._raw_paths:
            base_path: Path = self._raw_paths["base"]
            return base_path / test_image_path

        return test_image_path

    @cached_property
    def validation_set_image_paths(self) -> dict[str, Path]:
        """A dictionary of image paths to the extra validation sets."""
        match self.dataset_type:
            case _YoloDatasetType.DETECT:
                reserved_keys: set[str] = {
                    "path",
                    "train",
                    "val",
                    "test",
                    "names",
                }
            case _YoloDatasetType.POSE:
                reserved_keys: set[str] = {
                    "path",
                    "train",
                    "val",
                    "test",
                    "names",
                    "kpt_shape",
                    "kpt_names",
                }

        validation_set_keys: set[str] = set(self.raw_yaml_data.keys()) - reserved_keys

        validation_set_img_paths: dict[str, Path] = {}
        for k in validation_set_keys:
            if "base" in self._raw_paths:
                base_path: Path = self._raw_paths["base"]
                validation_set_img_paths[k] = base_path / Path(self.raw_yaml_data[k])
            else:
                validation_set_img_paths[k] = Path(self.raw_yaml_data[k])

        return validation_set_img_paths

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

    @cached_property
    def validation_set_label_paths(self) -> dict[str, Path]:
        """A dictionary of extra validation set paths."""
        return {
            k: p.parent.parent / "labels" / p.name
            for (k, p) in self.validation_set_image_paths.items()
        }

    @cached_property
    def names(self) -> list[str]:
        """A list of the names of bounding boxes."""
        return [
            self.raw_yaml_data["names"][sk]
            for sk in sorted(self.raw_yaml_data["names"].keys())
        ]

    @cached_property
    def kpt_names(self) -> list[str] | None:
        """A list of the names of keypoints."""
        if self.dataset_type != _YoloDatasetType.POSE:
            return None
        return [
            self.raw_yaml_data["kpt_names"][sk]
            for sk in sorted(self.raw_yaml_data["kpt_names"].keys())
        ]

    @cached_property
    def kpt_shape(self) -> KeypointShape | None:
        """A tuple containing the number of keypoints and 'dimensions' of each keypoint."""
        if self.dataset_type != _YoloDatasetType.POSE:
            return None

        if len(self.raw_yaml_data["kpt_shape"]) != 2:
            raise YoloDatasetError(
                "The shape of the 'kpt_shape' parameter in the yolo config is not 2. "
                f"Expected [num_keypoints, num_dimensions], got {self.raw_yaml_data['kpt_shape']}."
            )

        num_keypoints: int = self.raw_yaml_data["kpt_shape"][0]
        num_dimensions: int = self.raw_yaml_data["kpt_shape"][1]

        if not isinstance(num_keypoints, int):
            raise YoloDatasetError(
                f"Expected number of keypoints to be an integer, got {type(num_keypoints)!s}."
            )
        if not isinstance(num_dimensions, int):
            raise YoloDatasetError(
                "Expected number of keypoint dimensions to be an integer, got "
                f"{type(num_dimensions)!s}."
            )
        if num_dimensions not in [2, 3]:
            raise YoloDatasetError(
                f"Expected number of keypoint dimensions is 2 or 3, got {num_dimensions}."
            )

        if num_keypoints != len(self.kpt_names):
            raise YoloDatasetError(
                f"Number of keypoints in kpt_shape ({num_keypoints}) does not equal the number "
                f"of keypoints indicated by kpt_names ({len(self.kpt_names)})."
            )

        return KeypointShape(num_keypoints=num_keypoints, num_dimensions=num_dimensions)

    @classmethod
    def from_path(cls, path_to_dataset_yaml: Path) -> "_YoloDatasetConfig":
        """Creates a _YoloDatasetConfig object by reading data from a yaml file.

        Args:
            path_to_dataset_yaml (Path):
                The filepath to the dataset yaml file.

        Returns:
            A _YoloDatasetConfig object storing validated data from the dataset yaml file.
        """
        with open(path_to_dataset_yaml, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        return _YoloDatasetConfig(raw_yaml_data=data)


class _YoloDataset(BaseModel):
    """A class that reads a yolo dataset from disk into a VisionDataset.

    Attributes:
        config (_YoloDatasetConfig):
            The configuration for the yolo dataset.
    """

    model_config = ConfigDict(frozen=True)

    path_to_dataset: Path
    dataset_config: _YoloDatasetConfig

    @model_validator(mode="after")
    def validate_all_paths_exist(self) -> "_YoloDataset":
        """Validates that all the folder paths the should exist, do."""
        attributes_to_check: list[str] = [
            "train_images_path",
            "val_images_path",
            "train_labels_path",
            "val_labels_path",
        ]
        if self.dataset_config._raw_paths["test_images"] is not None:
            attributes_to_check += [
                "test_images_path",
                "test_labels_path",
            ]

        for attribute in attributes_to_check:
            path: Path = attrgetter(attribute)(self.dataset_config)
            if not path.exists():
                raise YoloDatasetError(
                    f'The "{attribute.replace("_", " ")}" is specified in the YOLO config file, '
                    + f"but doesn't exist. (path specified: {path})"
                )

        for (
            validation_set_image_path
        ) in self.dataset_config.validation_set_image_paths.values():
            if not validation_set_image_path.exists():
                raise YoloDatasetError(
                    f'The validation set path "images/{validation_set_image_path.name}" is '
                    f"specified in the YOLO config file, but doesn't exist. (path specified: {path})"
                )

        for (
            validation_set_label_path
        ) in self.dataset_config.validation_set_label_paths.values():
            if not validation_set_label_path.exists():
                raise YoloDatasetError(
                    f'The validation set path "labels/{validation_set_label_path.name}" is '
                    f"specified in the YOLO config file, but doesn't exist. (path specified: {path})"
                )
        return self

    @model_validator(mode="after")
    def validate_image_exists_for_all_annotations(self) -> "_YoloDataset":
        """Validates that for each"""
        return self

    @model_validator(mode="after")
    def warn_if_extra_dir_exists_in_images(self) -> "_YoloDatasetConfig":
        """Warns the user if an extra directory exists in the images folder not accounted for in the config."""
        dirs_accounted_for: set[str] = {"train", "val"} | set(
            self.dataset_config.validation_set_image_paths.keys()
        )
        if self.dataset_config.test_images_path is not None:
            dirs_accounted_for = dirs_accounted_for | {"test"}
        all_dirs_in_images_directory: set[str] = set(
            os.listdir(self.path_to_dataset / "images")
        )

        dirs_not_accounted_for: set[str] = (
            all_dirs_in_images_directory - dirs_accounted_for
        )

        if dirs_not_accounted_for:
            warnings.warn(
                (
                    "Directories exist in the dataset images directory not accounted for in the "
                    "dataset yaml config file.\n"
                    f"Directories not accounted for: {', '.join(sorted(dirs_not_accounted_for))}."
                ),
                YoloDatasetWarning,
            )

        return self

    @model_validator(mode="after")
    def warn_if_extra_dir_exists_in_labels(self) -> "_YoloDatasetConfig":
        """Warns the user if an extra directory exists in the labels folder not accounted for in the config."""
        dirs_accounted_for: set[str] = {"train", "val"} | set(
            self.dataset_config.validation_set_label_paths.keys()
        )
        if self.dataset_config.test_labels_path is not None:
            dirs_accounted_for = dirs_accounted_for | {"test"}

        all_dirs_in_labels_directory: set[str] = set(
            os.listdir(self.path_to_dataset / "labels")
        )

        dirs_not_accounted_for: set[str] = (
            all_dirs_in_labels_directory - dirs_accounted_for
        )

        if dirs_not_accounted_for:
            warnings.warn(
                (
                    "Directories exist in the dataset labels directory not accounted for in the "
                    "dataset yaml config file.\n"
                    f"Directories not accounted for: {', '.join(sorted(dirs_not_accounted_for))}."
                ),
                YoloDatasetWarning,
            )

        return self

    def read_yolo_annotation_file(
        self, path_to_yolo_annotation_file: Path
    ) -> list[Annotation]:
        """Reads a YOLO annotation file from disk.

        Args:
            path_to_yolo_annotation_file (Path):
                The path to the yolo annotation file on disk.

        Returns:
            The contents of the yolo annotation file as Annotation objects (Bounding boxes for
            detect datasets and bounding boxes with keypoints for pose datasets).
        """
        with open(path_to_yolo_annotation_file, "r") as f:
            lines: list[str] = f.readlines()

        match self.dataset_config.dataset_type:
            case _YoloDatasetType.DETECT:
                file_reading_fn: Callable[[list[str]], list[Annotation]] = (
                    self.read_yolo_detect_file
                )

            case _YoloDatasetType.POSE:
                file_reading_fn: Callable[[list[str]], list[Annotation]] = (
                    self.read_yolo_pose_file
                )

        try:
            return file_reading_fn(lines)
        except YoloDatasetError as yde:
            raise YoloDatasetError(
                "An exception has occured while reading file "
                f"{path_to_yolo_annotation_file.resolve()}.\n"
                f"{yde}"
            )

    def read_yolo_detect_file(
        self, yolo_detect_file_contents: list[str]
    ) -> list[BoundingBox]:
        """Reads a YOLO detect formatted annotations file."""
        bbox_annotations: list[BoundingBox] = []
        for line_ix, ann_line in enumerate(yolo_detect_file_contents):
            try:
                new_ann: BoundingBox = self.read_yolo_detect_annotation(ann_line)
                bbox_annotations.append(new_ann)
            except YoloDatasetError as yde:
                raise YoloDatasetError(
                    f"An exception has occured while reading line {line_ix + 1}.\n{yde}"
                )

        return bbox_annotations

    def read_yolo_detect_annotation(self, yolo_detect_annotation_line) -> BoundingBox:
        """Reads a YOLO detect formatted annotation line."""
        line_entries: list[str] = yolo_detect_annotation_line.strip().split(" ")
        if len(line_entries) < 5:
            raise YoloDatasetError(
                "Too few entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_detect_annotation_line}'."
            )
        if len(line_entries) > 5:
            raise YoloDatasetError(
                "Too many entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_detect_annotation_line}'."
            )

        class_index: int = int(line_entries[0])
        x_center: float = float(line_entries[1])
        y_center: float = float(line_entries[2])
        width: float = float(line_entries[3])
        height: float = float(line_entries[4])

        rect: Rectangle = Rectangle.from_center_xywh(x_center, y_center, width, height)

        if rect.left < 0:
            warnings.warn(
                f"Annotation left side is out of bounds: {rect}", YoloDatasetWarning
            )
        if rect.right > 1:
            warnings.warn(
                f"Annotation right side is out of bounds: {rect}", YoloDatasetWarning
            )
        if rect.top < 0:
            warnings.warn(
                f"Annotation top side is out of bounds: {rect}", YoloDatasetWarning
            )
        if rect.bottom > 1:
            warnings.warn(
                f"Annotation bottom side is out of bounds: {rect}", YoloDatasetWarning
            )

        if class_index < 0 or class_index > len(self.dataset_config.names) - 1:
            raise YoloDatasetError(
                "Class index is out of range for this dataset.\n"
                f"Class index read: {class_index}. Number of classes from the config "
                f"{len(self.dataset_config.names)}."
            )

        class_name: str = self.dataset_config.names[class_index]

        return BoundingBox(rectangle=rect, category=class_name)

    def read_yolo_pose_file(
        self, yolo_pose_file_contents: list[str]
    ) -> list[BoundingBoxWithKeypoints]:
        """Reads a YOLO pose formatted annotations file."""
        bbox_w_kpts_annotations: list[BoundingBoxWithKeypoints] = []
        for line_ix, ann_line in enumerate(yolo_pose_file_contents):
            try:
                new_ann: BoundingBoxWithKeypoints = self.read_yolo_pose_annotation(
                    ann_line
                )
                bbox_w_kpts_annotations.append(new_ann)
            except YoloDatasetError as yde:
                raise YoloDatasetError(
                    f"An exception has occured while reading line {line_ix + 1}.\n{yde}"
                )

        return bbox_w_kpts_annotations

    def read_yolo_pose_annotation(
        self, yolo_pose_annotation_line: str
    ) -> BoundingBoxWithKeypoints:
        """Reads a YOLO pose formatted annotation line."""
        line_entries: list[str] = yolo_pose_annotation_line.strip().split(" ")

        expected_number_of_line_entries: int = 5 + self.dataset_config.kpt_shape
        if len(line_entries) < 5:
            raise YoloDatasetError(
                "Too few entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_detect_annotation_line}'."
            )
        if len(line_entries) > 5:
            raise YoloDatasetError(
                "Too many entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_detect_annotation_line}'."
            )

        class_index: int = int(line_entries[0])
        x_center: float = float(line_entries[1])
        y_center: float = float(line_entries[2])
        width: float = float(line_entries[3])
        height: float = float(line_entries[4])
        raise NotImplementedError()

    @classmethod
    def from_folderpath(
        cls,
        folderpath: Path,
        yaml_file_name: str = "dataset.yaml",
    ) -> "_YoloDataset":
        """Creates a _YoloDataset from a folderpath.

        Args:
            folderpath (Path):
                The path to the root of the yolo dataset's folder.
            yaml_file_name (str):
                The filename of the dataset yaml config file.
                Defaults to 'dataset.yaml'.

        Returns:
            A _YoloDataset pointing to the yolo dataset on disk.
        """
        dataset_config: _YoloDatasetConfig = _YoloDatasetConfig.from_path(
            path_to_dataset_yaml=folderpath / yaml_file_name
        )
        yolo_dataset: _YoloDataset = _YoloDataset(
            path_to_dataset=folderpath,
            dataset_config=dataset_config,
        )
        return yolo_dataset


def read_yolo_dataset(
    folderpath: Path,
    yaml_file_name: str = "dataset.yaml",
) -> VisionDataset:
    """Reads a YOLO dataset from a path to the dataset's root folder.

    Args:
        folderpath (Path):
            The path to the root of the yolo dataset's folder.
        yaml_file_name (str):
            The filename of the dataset yaml config file.

    Returns:
        A VisionDataset containing the contents of the yolo dataset.
    """
    yolo_dataset: _YoloDataset = _YoloDataset.from_folderpath(
        folderpath=folderpath,
        yaml_file_name=yaml_file_name,
    )
