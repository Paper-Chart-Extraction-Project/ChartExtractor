"""A test module that tests the yolo_dataset module."""

import os
import textwrap
from pathlib import Path

import pytest
from dataset.adapters.yolo import YoloDatasetError, YoloDatasetWarning
from dataset.adapters.yolo.yolo_dataset import (
    _RawYamlData,
    _YoloDataset,
    _YoloDatasetConfig,
    _YoloDatasetType,
)

PATH_TO_TEST_DATA: Path = Path(__file__).parent / "data"

assert PATH_TO_TEST_DATA.exists()
assert PATH_TO_TEST_DATA.is_dir()


@pytest.fixture
def yaml_detect_data() -> _RawYamlData:
    """A dummy dictionary containing data found in a typical 'dataset.yaml' file.

    This dictionary is specifically for a 'detect' model.
    """
    return {
        "path": str(PATH_TO_TEST_DATA / "test_dataset"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["test_0", "test_1", "test_2"],
    }


@pytest.fixture
def yaml_pose_data() -> _RawYamlData:
    """A dummy dictionary containing data found in a typical 'dataset.yaml' file.

    This dictionary is specifically for a 'pose' model.
    """
    return {
        "path": str(PATH_TO_TEST_DATA / "test_dataset"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["test_0", "test_1", "test_2"],
        "kpt_shape": ["2", "2"],
        "kpt_names": ["left", "right"],
    }


class TestYoloDatasetConfig:
    """A collection of tests for the _YoloDatasetConfig class."""

    def test_all_necessary_keys_in_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that no error is raised when all necessary keys exist for a detect dataset."""
        _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)

    def test_necessary_key_missing_from_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that an error is raised when a necessary key doesn't exist in the raw yaml file."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data.pop("train")

        with pytest.raises(
            YoloDatasetError, match="don't exist in the yaml file: train"
        ):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_all_necessary_keys_missing_from_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that an error is raised when all the necessary keys don't exist in the raw yaml file."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data.pop("train")
        altered_yaml_detect_data.pop("val")
        altered_yaml_detect_data.pop("names")

        with pytest.raises(
            YoloDatasetError, match="don't exist in the yaml file: names, train, val."
        ):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_all_image_folderpath_parents_are_image(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that no error is raised when all the parents of the image folderpaths is 'image'."""
        _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)

    def test_one_image_folderpath_parent_isnt_image(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that an error is raised when one image folderpath has its parent != 'image'."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data["train"] = "image/train"

        with pytest.raises(YoloDatasetError, match="is not 'images'."):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_all_image_folderpaths_arent_image(self, yaml_detect_data: _RawYamlData):
        """Tests that an error is raised when all image folderpath parents aren't 'image'."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data["train"] = "image/train"
        altered_yaml_detect_data["val"] = "img/val"
        altered_yaml_detect_data["test"] = "imagess/test"

        with pytest.raises(YoloDatasetError, match="is not 'images'."):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_user_warned_when_relative_path_is_used(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that a warning is emitted when a relative path is found for 'path' in the dataset config."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data["path"] = "../test_dataset/"

        with pytest.warns(YoloDatasetWarning, match="is a relative path"):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_dataset_type_detect(self, yaml_detect_data: _RawYamlData):
        """Tests that the dataset type is set to 'detect' when a detection config is supplied."""
        detect_ds_config: _YoloDatasetConfig = _YoloDatasetConfig(
            raw_yaml_data=yaml_detect_data
        )

        assert detect_ds_config.dataset_type == _YoloDatasetType.DETECT

    def test_dataset_type_pose(self, yaml_pose_data: _RawYamlData):
        """Tests that the dataset type is set to 'pose' when a pose dataset is supplied."""
        pose_ds_config: _YoloDatasetConfig = _YoloDatasetConfig(
            raw_yaml_data=yaml_pose_data
        )

        assert pose_ds_config.dataset_type == _YoloDatasetType.POSE

    def test_dataset_type_pose_but_no_kpt_shape_is_supplied(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that the dataset type is set to 'pose' when a yaml file with 'kpt_shape' is missing."""
        altered_yaml_pose_data: _RawYamlData = yaml_pose_data
        altered_yaml_pose_data.pop("kpt_shape")

        with pytest.raises(KeyError, match="'kpt_names' but not 'kpt_shape'"):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_pose_data).dataset_type  # noqa: B018

    def test_dataset_type_pose_but_no_kpt_names_is_supplied(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that the dataset type is set to 'pose' when a yaml file with 'kpt_names' is missing."""
        altered_yaml_pose_data: _RawYamlData = yaml_pose_data
        altered_yaml_pose_data.pop("kpt_names")

        with pytest.raises(KeyError, match="'kpt_shape' but not 'kpt_names'"):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_pose_data).dataset_type  # noqa: B018

    def test_labels_path_constructs_properly(self, yaml_pose_data: _RawYamlData):
        """Tests that the labels path is constructed by just swapping the 'images' folder for 'labels'."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_pose_data)
        assert (
            config.train_labels_path == config._raw_paths["base"] / "labels" / "train"
        )

    def test_extra_validation_set_paths(self, yaml_detect_data: _RawYamlData):
        """Tests that extra validation set paths get added to the 'validation_set_image_paths' attribute."""
        altered_yaml_pose_data: _RawYamlData = yaml_detect_data
        altered_yaml_pose_data["test_2"] = "images/test_2"
        altered_yaml_pose_data.pop("path")
        config: _YoloDatasetType = _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)

        assert config.validation_set_image_paths == {"test_2": Path("images/test_2")}
        assert config.validation_set_label_paths == {"test_2": Path("labels/test_2")}


class TestYoloDataset:
    """A collection of tests for the _YoloDataset class."""

    def test_valid_folder_structure(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset with a valid folder structure."""
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

        _YoloDataset.from_folderpath(tmp_path / "test_dataset_valid_folder_structure")

    def test_invalid_folder_structure_missing_images_folder(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset with a missing split images folder."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "labels",
            Path("test_dataset_valid_folder_structure") / "labels" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "val",
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

        with pytest.raises(YoloDatasetError, match="but doesn't exist."):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_labels_folder(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset with a missing split labels folder."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "images" / "train",
            Path("test_dataset_valid_folder_structure") / "images" / "val",
            Path("test_dataset_valid_folder_structure") / "images" / "test",
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

        with pytest.raises(YoloDatasetError, match="but doesn't exist."):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_split(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset where an entire split is missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_split_images(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset where a split's images are missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
            Path("test_dataset_valid_folder_structure") / "images" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "train",
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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_split_labels(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset where an split's labels are missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
            Path("test_dataset_valid_folder_structure") / "images" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "train",
            Path("test_dataset_valid_folder_structure") / "images" / "val",
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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_validation_set(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset where an entire validation set is missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
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
            test_2: images/test2

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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_validation_set_images(
        self, tmp_path: Path
    ):
        """Tests the construction of a _YoloDataset where a validation set's images are missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
            Path("test_dataset_valid_folder_structure") / "images" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "val",
            Path("test_dataset_valid_folder_structure") / "images" / "test",
            Path("test_dataset_valid_folder_structure") / "labels" / "test",
            Path("test_dataset_valid_folder_structure") / "labels" / "test2",
        ]
        yaml_data: str = textwrap.dedent(f"""
            path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
            train: images/train
            val: images/val
            test: images/test
            test_2: images/test2

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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_invalid_folder_structure_missing_validation_set_labels(
        self, tmp_path: Path
    ):
        """Tests the construction of a _YoloDataset where an validation set's labels are missing."""
        dirs_to_create: list[Path] = [
            Path("test_dataset_valid_folder_structure"),
            Path("test_dataset_valid_folder_structure") / "images",
            Path("test_dataset_valid_folder_structure") / "labels",
            Path("test_dataset_valid_folder_structure") / "images" / "train",
            Path("test_dataset_valid_folder_structure") / "labels" / "train",
            Path("test_dataset_valid_folder_structure") / "images" / "val",
            Path("test_dataset_valid_folder_structure") / "images" / "test",
            Path("test_dataset_valid_folder_structure") / "labels" / "test",
            Path("test_dataset_valid_folder_structure") / "labels" / "test2",
        ]
        yaml_data: str = textwrap.dedent(f"""
            path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
            train: images/train
            val: images/val
            test: images/test
            test_2: images/test2

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

        with pytest.raises(YoloDatasetError, match="but doesn't exist"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_valid_folder_structure_extra_images_folder(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset with an extra images folder that is unaccounted for."""
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
            Path("test_dataset_valid_folder_structure") / "images" / "test2",
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

        with pytest.warns(YoloDatasetWarning, match="not accounted for"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_valid_folder_structure_extra_labels_folder(self, tmp_path: Path):
        """Tests the construction of a _YoloDataset with an extra images folder that is unaccounted for."""
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
            Path("test_dataset_valid_folder_structure") / "labels" / "test2",
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

        with pytest.warns(YoloDatasetWarning, match="not accounted for"):
            _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )
