"""A module with functions that read a YoloDataset from disk into a VisionDataset."""

import warnings
from collections.abc import Callable
from itertools import batched
from pathlib import Path

from core.annotations import (
    Annotation,
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point, Rectangle
from image import ImageMetadata

from dataset import AnnotatedImage, ImageId, VisionDataset

from .exceptions import YoloDatasetError
from .warnings import YoloDatasetWarning
from .yolo_dataset import _YoloDataset, _YoloDatasetConfig, _YoloDatasetType


def read_yolo_dataset(
    folderpath: Path | str,
    yaml_file_name: str = "dataset.yaml",
) -> VisionDataset:
    """Reads a YOLO dataset from a path to the dataset's root folder.

    Args:
        folderpath (Path | str):
            The path to the root of the yolo dataset's folder.
            If a string is provided, it is cast to a Path.
        yaml_file_name (str):
            The filename of the dataset yaml config file.

    Returns:
        A VisionDataset containing the contents of the yolo dataset.
    """
    if isinstance(folderpath, str):
        folderpath: Path = Path(folderpath)

    yolo_dataset: _YoloDataset = _YoloDataset.from_folderpath(
        folderpath=folderpath,
        yaml_file_name=yaml_file_name,
    )
    return convert_yolo_dataset_to_vision_dataset(yolo_dataset)


def convert_yolo_dataset_to_vision_dataset(yolo_dataset: _YoloDataset) -> VisionDataset:
    """Reads a yolo dataset from disk into a VisionDataset.

    Args:
        yolo_dataset (_YoloDataset):
            The yolo dataset to convert to a vision dataset.

    Returns:
        A VisionDataset with the contents of the yolo dataset from disk.
    """
    images: dict[str, dict[ImageId, ImageMetadata]] = {
        split: {
            im_path.stem: ImageMetadata(path=im_path) for im_path in split_image_paths
        }
        for split, split_image_paths in yolo_dataset.images.items()
    }
    annotations: dict[str, dict[ImageId, list[Annotation]]] = {
        split: {
            ann_path.stem: yolo_dataset.read_yolo_annotation_file(
                path_to_yolo_annotation_file=ann_path
            )
            for ann_path in split_annotation_paths
        }
        for split, split_annotation_paths in yolo_dataset.annotations.items()
    }

    annotated_images: list[AnnotatedImage] = []
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

    return VisionDataset(
        annotated_images=annotated_images,
        dataset_metadata={},  # todo: dump the config to a dictionary.
        splits=splits,
        evaluation_sets=extra_val_sets,
    )


def read_yolo_annotation_file(
    dataset_config: _YoloDatasetConfig,
    path_to_yolo_annotation_file: Path,
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

    match dataset_config.dataset_type:
        case _YoloDatasetType.DETECT:
            file_reading_fn: Callable[[list[str]], list[Annotation]] = (
                read_yolo_detect_file
            )

        case _YoloDatasetType.POSE:
            file_reading_fn: Callable[[list[str]], list[Annotation]] = (
                read_yolo_pose_file
            )

    try:
        return file_reading_fn(lines)
    except YoloDatasetError as yde:
        raise YoloDatasetError(
            "An exception has occured while reading file "
            f"{path_to_yolo_annotation_file.resolve()}.\n"
            f"{yde}"
        )


def read_yolo_detect_file(yolo_detect_file_contents: list[str]) -> list[BoundingBox]:
    """Reads a YOLO detect formatted annotations file.

    Args:
        yolo_detect_file_contents (list[str]):
            The contents of the yolo file read into lines.

    Returns:
        A list of bounding boxes, one for each line in the file.
    """
    bbox_annotations: list[BoundingBox] = []
    for line_ix, ann_line in enumerate(yolo_detect_file_contents):
        try:
            new_ann: BoundingBox = read_yolo_detect_annotation(ann_line)
            bbox_annotations.append(new_ann)
        except YoloDatasetError as yde:
            raise YoloDatasetError(
                f"An exception has occured while reading line {line_ix + 1}.\n"
                f"Full line: '{ann_line}'\n"
                f"{yde}"
            )

    return bbox_annotations


def read_yolo_detect_annotation(
    yolo_detect_annotation_line: str,
    class_names: list[str],
) -> BoundingBox:
    """Parses a YOLO detect formatted annotation line into a BoundingBox.

    Yolo detect annotations have five components.
    The formatting for each line is:
    '{class id} {x} {y} {width} {height}'
    Where
    'class id' is a number denoting the class of the object (the name for each id can be found
    in the class_names list).
    'x' is the x coordinate for the center of the bounding box, normalized by image width.
    'y' is the y coordinate for the center of the bounding box, normalized by image height.
    'width' is the distance from the left side of the bounding box to the right side, normalized by
    image width.
    'height' is the distance from the top side of the bounding box to the bottom side, normalized
    by image height.


    Args:
        yolo_detect_annotation_line (str):
            The line from a yolo annotation file in the 'detect' format.
            That is, only a class id and bounding box data.
        class_names (list[str]):
            The class names for the dataset.
            The first value in each line corresponds to the index its class name in the class
            names array.
            For example, the line '3 0.5 0.5 0.25 0.3' would have a class name at index 3
            in the class_names array.

    Returns:
        A BoundingBox object containing the data of the annotation line string.
    """
    expected_number_of_entries: int = 5
    line_entries: list[str] = yolo_detect_annotation_line.strip().split(" ")
    if len(line_entries) != expected_number_of_entries:
        raise YoloDatasetError(
            "Incorrect number of entries in detect annotation line. "
            f"Got {len(line_entries)} when 5 were expected.\n"
        )

    try:
        class_index: int = int(line_entries[0])
        x_center: float = float(line_entries[1])
        y_center: float = float(line_entries[2])
        width: float = float(line_entries[3])
        height: float = float(line_entries[4])
    except ValueError as e:
        raise YoloDatasetError(
            f"Could not parse entry in yolo detect annotation line.\n{e}"
        )

    if any(entry < 0 or entry > 1 for entry in [x_center, y_center, width, height]):
        raise YoloDatasetError(
            "Entry in yolo detect annotation is not normalized to [0, 1].\n"
        )

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

    if class_index < 0 or class_index > len(class_names) - 1:
        raise YoloDatasetError(
            "Class index is out of range for this dataset.\n"
            f"Class index read: {class_index}. Number of classes from the config "
            f"{len(class_names)}."
        )

    class_name: str = class_names[class_index]

    return BoundingBox(rectangle=rect, category=class_name)


def read_yolo_pose_file(
    self, yolo_pose_file_contents: list[str]
) -> list[BoundingBoxWithKeypoints]:
    """Reads a YOLO pose formatted annotations file."""
    bbox_w_kpts_annotations: list[BoundingBoxWithKeypoints] = []
    for line_ix, ann_line in enumerate(yolo_pose_file_contents):
        try:
            new_ann: BoundingBoxWithKeypoints = self.read_yolo_pose_annotation(ann_line)
            bbox_w_kpts_annotations.append(new_ann)
        except YoloDatasetError as yde:
            raise YoloDatasetError(
                f"An exception has occured while reading line {line_ix + 1}.\n"
                f"Full line: '{ann_line}'\n"
                f"{yde}"
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
