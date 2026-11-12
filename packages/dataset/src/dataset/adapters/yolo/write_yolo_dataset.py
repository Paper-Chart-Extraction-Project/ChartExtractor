"""A module with functions to write a VisionDataset onto disk as a yolo dataset."""

import warnings
from pathlib import Path

from core.annotations import (
    Annotation,
    BoundingBox,
    BoundingBoxWithKeypoints,
    VisibilityStatus,
)
from image import Image, ImageMetadata

from dataset import AnnotatedImage, ImageId, VisionDataset

from .exceptions import YoloDatasetError
from .warnings import YoloDatasetWarning


def write_vision_dataset_to_disk(dataset: VisionDataset, folderpath: Path) -> None:
    """Writes a vision dataset to disk using this YoloDataset's config.

    Args:
        dataset (VisionDataset):
            The vision dataset to write to disk.
        folderpath (Path):
            The path to the base folder where the yolo dataset will be written.
    """
    create_directory_structure(dataset, folderpath)
    write_dataset_yaml_to_disk(dataset, folderpath)

    im_id_to_annotated_image_map: dict[ImageId, AnnotatedImage] = {
        ann_im.image_id: ann_im for ann_im in dataset.annotated_images
    }
    splits_and_evaluation_sets: dict[str, set[ImageId]] = (
        dataset.splits | dataset.evaluation_sets
    )
    for image_set, image_ids_in_set in splits_and_evaluation_sets.items():
        for im_id in image_ids_in_set:
            write_image_to_disk(
                image=im_id_to_annotated_image_map[im_id].image,
                filepath=folderpath / "images" / image_set / f"{im_id}.jpg",
            )
            write_annotations_to_disk(
                annotations=im_id_to_annotated_image_map[im_id].annotations,
                filepath=folderpath / "labels" / image_set / f"{im_id}.txt",
            )


def create_directory_structure(dataset: VisionDataset, folderpath: Path) -> None:
    """Creates the folders for writing yolo dataset files into.

    Args:
        dataset (VisionDataset):
            The vision dataset to create the directory structure for.
        folderpath (Path):
            The path to the base folder where the yolo dataset will be written.
    """
    splits_and_evaluation_set_names: list[str] = list(
        (dataset.splits | dataset.evaluation_sets).keys()
    )
    for set_name in splits_and_evaluation_set_names:
        Path(folderpath / "images" / set_name).mkdir()
        Path(folderpath / "labels" / set_name).mkdir()


def write_dataset_yaml_to_disk(
    dataset: VisionDataset, folderpath: Path, indent: int = 2
) -> None:
    """Creates the dataset.yaml config file.

    Args:
        dataset (VisionDataset):
            The vision dataset to create the directory structure for.
        folderpath (Path):
            The path to the base folder where the dataset yaml file will be written.
        indent (int):
            The number of spaces to indent lines.
    """

    yaml_lines: list[str] = [
        f"path: {folderpath.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in dataset.splits:
        yaml_lines.append("test: images/test")

    yaml_lines.append("\n")

    yaml_lines.append("names:")
    names: list[str] = get_names_from_vision_dataset(dataset)
    for ix, name in enumerate(names):
        yaml_lines.append(" " * indent + f"{ix}: {name}")

    if any(
        isinstance(ann, BoundingBoxWithKeypoints)
        for ann in [
            x
            for y in [ann_im.annotations for ann_im in dataset.annotated_images]
            for x in y
        ]
    ):
        generate_pose_lines_for_yaml(dataset)


def get_names_from_vision_dataset(
    dataset: VisionDataset, strict: bool = True
) -> list[str]:
    """Gets the names of all the bounding boxes from a vision dataset.

    Args:
        dataset (VisionDataset):
            The vision dataset to get names from.
        strict (bool):
            Whether or not unsupported types should raise an error.
    """
    names: list[str] = []
    for ann_im in dataset.annotated_images:
        for annotation in ann_im.annotations:
            match annotation:
                case BoundingBox():
                    names.append(annotation.category)
                case BoundingBoxWithKeypoints():
                    names.append(annotation.bounding_box.category)
                case _:
                    if strict:
                        raise YoloDatasetError(
                            "An error occured while writing vision dataset.\n"
                            f"Cannot get the class name for type {type(annotation)}."
                        )
                    else:
                        warnings.warn(
                            f"Cannot get the class name for type {type(annotation)}.",
                            YoloDatasetWarning,
                        )

    return sorted(set(names))


def generate_pose_lines_for_yaml(dataset: VisionDataset):
    """Adds special lines to the dataset yaml file for pose estimation."""
    flatten = lambda l: [x for y in l for x in y]
    all_visibility_values_in_dataset: list[VisibilityStatus] = [
        [
            [kpt.visibility_status for kpt in annotation.keypoints]
            for annotation in ann_im.annotations
        ]
        for ann_im in dataset.annotated_images
    ]
    all_visibility_values_in_dataset: list[VisibilityStatus] = flatten(
        flatten(all_visibility_values_in_dataset)
    )
    kpt_visibility: int = (
        2 if set(all_visibility_values_in_dataset == {VisibilityStatus.VISIBLE}) else 3
    )
    # kpt_shape: int =
    raise NotImplementedError()


def write_image_to_disk(image: ImageMetadata | Image, filepath: Path) -> None:
    """Writes an image to disk.

    Args:
        image (ImageMetadata | Image):
            The image to write to disk.
        filepath (Path):
            The path to the file on disk to write the image to.
            The file must not exist before calling this method.
    """
    raise NotImplementedError()


def write_annotations_to_disk(annotations: list[Annotation], filepath: Path) -> None:
    """Writes annotations to disk using this YoloDataset's config.

    Args:
        annotations (list[Annotations]):
            The annotations to write.
        filepath (Path):
            The path to the file on disk to write the annotations to.
            The file must not exist before calling this method.
    """
    raise NotImplementedError()
