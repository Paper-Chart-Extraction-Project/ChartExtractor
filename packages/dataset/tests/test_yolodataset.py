"""A test module that tests the yolo_dataset module."""

from pathlib import Path

import pytest
from dataset.adapters.yolo import YoloDatasetError, YoloDatasetWarning
from dataset.adapters.yolo.yolo_dataset import (
    _RawYamlData,
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
    """A test class that tests the _YoloDatasetConfig class."""

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
