"""A module with test fixtures shared between files."""

import textwrap
from pathlib import Path

import pytest

from dataset.adapters.yolo.yolo_dataset import _RawYamlData, _YoloDataset


@pytest.fixture
def yaml_detect_data(tmp_path: Path) -> _RawYamlData:
    """A dummy dictionary containing data found in a typical 'dataset.yaml' file.

    This dictionary is specifically for a 'detect' model.
    """
    return {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "test_0", 1: "test_1", 2: "test_2"},
    }


@pytest.fixture
def yaml_pose_data(tmp_path: Path) -> _RawYamlData:
    """A dummy dictionary containing data found in a typical 'dataset.yaml' file.

    This dictionary is specifically for a 'pose' model.
    """
    return {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "test_0", 1: "test_1", 2: "test_2"},
        "kpt_shape": [2, 2],
        "kpt_names": {0: ["left", "right"], 1: ["one", "two"]},
    }


@pytest.fixture
def yolo_detect_dataset(tmp_path: Path) -> _YoloDataset:
    """A dummy detect dataset."""
    dirs_to_create: list[Path] = [
        Path("test_dataset_valid_folder_structure"),
        Path("test_dataset_valid_folder_structure") / "images",
        Path("test_dataset_valid_folder_structure") / "labels",
        Path("test_dataset_valid_folder_structure") / "images" / "train",
        Path("test_dataset_valid_folder_structure") / "labels" / "train",
        Path("test_dataset_valid_folder_structure") / "images" / "val",
        Path("test_dataset_valid_folder_structure") / "labels" / "val",
        Path("test_dataset_valid_folder_structure") / "images" / "test",
        Path("test_dataset_valid_folder_structure") / "labels" / "test",
    ]
    yaml_data: str = textwrap.dedent(f"""
        path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
        train: images/train
        val: images/val
        test: images/test

        names:
            0: test_0
            1: test_1       
    """).strip()
    for dir in dirs_to_create:
        (tmp_path / dir).mkdir()
    dataset_yaml: Path = (
        tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
    )
    dataset_yaml.write_text(yaml_data)

    return _YoloDataset.from_folderpath(
        tmp_path / "test_dataset_valid_folder_structure"
    )


@pytest.fixture
def yolo_pose_dataset_no_vis(tmp_path: Path) -> _YoloDataset:
    """A dummy pose dataset."""
    dirs_to_create: list[Path] = [
        Path("test_dataset_valid_folder_structure"),
        Path("test_dataset_valid_folder_structure") / "images",
        Path("test_dataset_valid_folder_structure") / "labels",
        Path("test_dataset_valid_folder_structure") / "images" / "train",
        Path("test_dataset_valid_folder_structure") / "labels" / "train",
        Path("test_dataset_valid_folder_structure") / "images" / "val",
        Path("test_dataset_valid_folder_structure") / "labels" / "val",
        Path("test_dataset_valid_folder_structure") / "images" / "test",
        Path("test_dataset_valid_folder_structure") / "labels" / "test",
    ]
    yaml_data: str = textwrap.dedent(f"""
        path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
        train: images/train
        val: images/val
        test: images/test

        kpt_shape: [2, 2]

        kpt_names:
            0:
                - kpt_1
                - kpt_2
            1:
                - kpt_a
                - kpt_b
        
        names:
            0: test_0
            1: test_1
    """).strip()
    for dir in dirs_to_create:
        (tmp_path / dir).mkdir()
    dataset_yaml: Path = (
        tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
    )
    dataset_yaml.write_text(yaml_data)

    return _YoloDataset.from_folderpath(
        tmp_path / "test_dataset_valid_folder_structure"
    )


@pytest.fixture
def yolo_pose_dataset_with_vis(tmp_path: Path) -> _YoloDataset:
    """A dummy pose dataset."""
    dirs_to_create: list[Path] = [
        Path("test_dataset_valid_folder_structure"),
        Path("test_dataset_valid_folder_structure") / "images",
        Path("test_dataset_valid_folder_structure") / "labels",
        Path("test_dataset_valid_folder_structure") / "images" / "train",
        Path("test_dataset_valid_folder_structure") / "labels" / "train",
        Path("test_dataset_valid_folder_structure") / "images" / "val",
        Path("test_dataset_valid_folder_structure") / "labels" / "val",
        Path("test_dataset_valid_folder_structure") / "images" / "test",
        Path("test_dataset_valid_folder_structure") / "labels" / "test",
    ]
    yaml_data: str = textwrap.dedent(f"""
        path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
        train: images/train
        val: images/val
        test: images/test

        kpt_shape: [2, 3]

        kpt_names:
            0:
                - kpt_1
                - kpt_2
            1:
                - kpt_a
                - kpt_b
        
        names:
            0: test_0
            1: test_1
    """).strip()
    for dir in dirs_to_create:
        (tmp_path / dir).mkdir()
    dataset_yaml: Path = (
        tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
    )
    dataset_yaml.write_text(yaml_data)

    return _YoloDataset.from_folderpath(
        tmp_path / "test_dataset_valid_folder_structure"
    )
