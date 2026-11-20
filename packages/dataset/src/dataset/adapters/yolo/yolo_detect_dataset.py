"""A module that can read/write yolo detect datasets to DetectVisionDataset."""

import warnings
from pathlib import Path
from typing import cast

from core.annotations import Annotation, BoundingBox
from core.geometry import Rectangle
from image import ImageMetadata

from dataset import AnnotatedImage, DetectVisionDataset, ImageId, Splits

from .exceptions import YoloDatasetError
from .warnings import YoloDatasetWarning
from .yolo_dataset import _YoloDataset, _YoloDatasetConfig


class _YoloDetectDatasetConfig(_YoloDatasetConfig):
    """Reads, validates, and stores data from a yolo detect dataset.yaml file.

    Identical to the base class, but exists for typing reasons.
    """


class _YoloDetectDataset(_YoloDataset):
    """A class that reads a yolo detect dataset from disk into a DetectVisionDataset.

    Attributes:
        config (_YoloDetectDatasetConfig):
            The configuration for the yolo detect dataset.
    """

    dataset_config: _YoloDetectDatasetConfig

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
            return self.read_yolo_detect_file(lines)  # pyright: ignore
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

    @classmethod
    def from_folderpath(
        cls,
        folderpath: Path,
        yaml_file_name: str = "dataset.yaml",
    ) -> _YoloDataset:
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
        dataset_config: _YoloDetectDatasetConfig = _YoloDetectDatasetConfig.from_path(
            path_to_dataset_yaml=folderpath / yaml_file_name
        )
        yolo_dataset: _YoloDataset = _YoloDetectDataset(
            path_to_dataset=folderpath,
            dataset_config=dataset_config,
        )
        return yolo_dataset

    def read_to_vision_dataset(self) -> DetectVisionDataset:
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
        annotations: dict[str, dict[ImageId, list[Annotation]]] = {
            split: {
                ann_path.stem: self.read_yolo_annotation_file(
                    path_to_yolo_annotation_file=ann_path
                )
                for ann_path in split_annotation_paths
            }
            for split, split_annotation_paths in self.annotations.items()
        }

        annotated_images: list[AnnotatedImage[BoundingBox]] = []
        for split, image_id_to_image_metadata_map in images.items():
            for img_id, image_metadata in image_id_to_image_metadata_map.items():
                annotated_image: AnnotatedImage = AnnotatedImage(
                    image=image_metadata,
                    annotations=annotations[split][img_id],
                    image_id=img_id,
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

        return DetectVisionDataset(
            annotated_images=annotated_images,
            dataset_metadata={},  # todo: dump the config to a dictionary.
            splits=cast(Splits, splits),
            evaluation_sets=extra_val_sets,
        )
