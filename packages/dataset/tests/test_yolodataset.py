"""A test module that tests the yolo_dataset module."""

from dataset.adapters.yolo.yolo_dataset import _YoloDatasetConfig, _RawYamlData

from pathlib import Path
import pytest

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

    def test_all_necessary_keys_in_raw_yaml_data(self):
        """Tests that no error is raised when all necessary keys exist."""

    def test_necessary_key_missing_from_raw_yaml_data(self):
        """Tests that an error is raised when a necessary key doesn't exist in the raw yaml file."""

    def test_all_necessary_keys_missing_from_raw_yaml_data(self):
        """Tests that an error is raised when all the necessary keys don't exist in the raw yaml file."""

    def test_all_image_folderpath_parents_are_image(self):
        """Tests that no error is raised when all the parents of the image folderpaths is 'image'."""

    def test_one_image_folderpath_parent_isnt_image(self):
        """Tests that an error is raised when one image folderpath has its parent != 'image'."""

    def test_all_image_folderpaths_arent_image(self):
        """Tests that an error is raised when all image folderpath parents aren't 'image'."""

    def test_user_warned_when_relative_path_is_used(self):
        """Tests that a warning is emitted when a relative path is found for 'path' in the dataset config."""

    def test_dataset_type_detect(self):
        """Tests that the dataset type is set to 'detect' when a detection config is supplied."""

    def test_dataset_type_pose(self):
        """Tests that the dataset type is set to 'pose' when a pose dataset is supplied."""
