"""A module that can read/write yolo pose datasets to PoseVisionDataset."""

from collections import namedtuple
from functools import cached_property
from itertools import batched
from pathlib import Path
from typing import Self, cast

from core.annotations import (
    Annotation,
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point
from image import ImageMetadata
from pydantic import model_validator

from dataset import AnnotatedImage, ImageId, PoseVisionDataset, Splits

from .exceptions import YoloDatasetConfigError, YoloDatasetError
from .yolo_detect_dataset import _YoloDetectDataset, _YoloDetectDatasetConfig

KeypointShape = namedtuple("KeypointShape", ["num_keypoints", "num_dimensions"])


class _YoloPoseDatasetConfig(_YoloDetectDatasetConfig):
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


class _YoloPoseDataset(_YoloDetectDataset):
    """A class that reads a yolo pose dataset from disk into a PoseVisionDataset."""

    dataset_config: _YoloPoseDatasetConfig

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

        try:
            return self.read_yolo_pose_file(lines)  # pyright: ignore
        except YoloDatasetError as yde:
            raise YoloDatasetError(
                "An exception has occured while reading file "
                f"{path_to_yolo_annotation_file.resolve()}.\n"
                f"{yde}"
            )

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

        expected_number_of_line_entries: int = 5 + (
            self.dataset_config.kpt_shape.num_keypoints
            * self.dataset_config.kpt_shape.num_dimensions
        )
        if len(line_entries) < expected_number_of_line_entries:
            raise YoloDatasetError(
                "Too few entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_pose_annotation_line}'."
            )
        if len(line_entries) > expected_number_of_line_entries:
            raise YoloDatasetError(
                "Too many entries in detect annotation line. "
                f"Got {len(line_entries)} when 5 were expected.\n"
                f"Full line: '{yolo_pose_annotation_line}'."
            )

        class_index: int = int(line_entries[0])
        x_center: float = float(line_entries[1])
        y_center: float = float(line_entries[2])
        width: float = float(line_entries[3])
        height: float = float(line_entries[4])

        bbox: BoundingBox = self.read_yolo_detect_annotation(
            f"{class_index} {x_center} {y_center} {width} {height}"
        )

        keypoints: list[Keypoint] = []
        if self.dataset_config.kpt_shape.num_dimensions == 2:
            for kpt_ix, (kpt_x, kpt_y) in enumerate(batched(line_entries[5:], 2)):
                vis_status: VisibilityStatus = VisibilityStatus.VISIBLE
                kpt_name: str = self.dataset_config.kpt_names[class_index][kpt_ix]
                keypoint: Keypoint = Keypoint(
                    point=Point(x=float(kpt_x), y=float(kpt_y)),
                    category=kpt_name,
                    visibility=vis_status,
                )
                keypoints.append(keypoint)

        else:
            for kpt_ix, (kpt_x, kpt_y, visibility) in enumerate(
                batched(line_entries[5:], 3)
            ):
                match visibility:
                    case "0":
                        vis_status: VisibilityStatus = VisibilityStatus.MISSING
                    case "1":
                        vis_status: VisibilityStatus = VisibilityStatus.OCCLUDED
                    case "2":
                        vis_status: VisibilityStatus = VisibilityStatus.VISIBLE
                    case _:
                        raise YoloDatasetError(
                            f"Visibility status ({visibility}) for keypoint {kpt_ix} is not in "
                            "[0, 1, 2]."
                        )
                kpt_name: str = self.dataset_config.kpt_names[class_index][kpt_ix]
                print(f"Keypoint name: {kpt_name}")
                keypoint: Keypoint = Keypoint(
                    point=Point(x=float(kpt_x), y=float(kpt_y)),
                    category=kpt_name,
                    visibility=vis_status,
                )
                keypoints.append(keypoint)

        return BoundingBoxWithKeypoints(bounding_box=bbox, keypoints=keypoints)

    @classmethod
    def from_folderpath(
        cls,
        folderpath: Path,
        yaml_file_name: str = "dataset.yaml",
    ) -> Self:
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
        dataset_config: _YoloPoseDatasetConfig = _YoloPoseDatasetConfig.from_path(
            path_to_dataset_yaml=folderpath / yaml_file_name
        )
        yolo_dataset: Self = cls(
            path_to_dataset=folderpath,
            dataset_config=dataset_config,
        )
        return yolo_dataset

    def read_to_vision_dataset(self) -> PoseVisionDataset:
        """Reads a yolo dataset from disk into a VisionDataset.

        Returns:
            A VisionDataset with the contents of the yolo dataset from disk.
        """
        images: dict[str, dict[ImageId, ImageMetadata]] = {
            split: {
                im_path.stem: ImageMetadata(path=im_path)
                for im_path in split_image_paths
            }
            for split, split_image_paths in self.images.items()
        }
        annotations: dict[str, dict[ImageId, list[BoundingBoxWithKeypoints]]] = {
            split: {
                ann_path.stem: self.read_yolo_annotation_file(
                    path_to_yolo_annotation_file=ann_path
                )
                for ann_path in split_annotation_paths
            }
            for split, split_annotation_paths in self.annotations.items()
        }

        annotated_images: list[AnnotatedImage[BoundingBoxWithKeypoints]] = []
        for split, image_id_to_image_metadata_map in images.items():
            for img_id, image_metadata in image_id_to_image_metadata_map.items():
                annotated_image: AnnotatedImage[BoundingBoxWithKeypoints] = (
                    AnnotatedImage[BoundingBoxWithKeypoints](
                        image=image_metadata,
                        annotations=annotations[split][img_id],
                        image_id=img_id,
                    )
                )
                annotated_images.append(annotated_image)

        splits: dict[str, set[ImageId]] = {
            split: set(images[split].keys())
            for split in ("train", "val", "test")
            if split in images
        }
        extra_val_sets: dict[str, set[ImageId]] = {
            split: set(images[split].keys())
            for split in sorted(set(images.keys()) - {"train", "val", "test"})
        }

        return PoseVisionDataset(
            annotated_images=annotated_images,
            dataset_metadata={},  # todo: dump the config to a dictionary.
            splits=cast(Splits, splits),
            evaluation_sets=extra_val_sets,
        )
