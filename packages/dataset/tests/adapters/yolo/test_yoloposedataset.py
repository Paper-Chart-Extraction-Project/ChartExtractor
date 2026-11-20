"""A test module that tests the yolo_pose_dataset module."""

import pytest

from dataset.adapters.yolo import YoloDatasetConfigError
from dataset.adapters.yolo.yolo_dataset import _RawYamlData
from dataset.adapters.yolo.yolo_pose_dataset import _YoloPoseDatasetConfig


class TestYoloPoseDatasetConfig:
    """A collection of tests for the _YoloPoseDatasetConfig class."""

    def test_all_necessary_keys_in_raw_pose_yaml_data(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that no error is raised when all necessary keys exist for a pose dataset."""
        _YoloPoseDatasetConfig(raw_yaml_data=yaml_pose_data)

    def test_dataset_type_pose_but_no_kpt_shape_is_supplied(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that the dataset type is set to 'pose' when a yaml file with 'kpt_shape' is missing."""
        altered_yaml_pose_data: _RawYamlData = yaml_pose_data
        altered_yaml_pose_data.pop("kpt_shape")

        with pytest.raises(YoloDatasetConfigError, match="kpt_shape"):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_pose_data)

    def test_dataset_type_pose_but_no_kpt_names_is_supplied(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that the dataset type is set to 'pose' when a yaml file with 'kpt_names' is missing."""
        altered_yaml_pose_data: _RawYamlData = yaml_pose_data
        altered_yaml_pose_data.pop("kpt_names")

        with pytest.raises(YoloDatasetConfigError, match="kpt_names"):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_pose_data)

    def test_kpt_names_is_a_sorted_list(self, yaml_pose_data: _RawYamlData):
        """Tests that the names list is created from the dictionary and sorts properly."""
        config: _YoloPoseDatasetConfig = _YoloPoseDatasetConfig(
            raw_yaml_data=yaml_pose_data
        )
        assert config.kpt_names == [["left", "right"], ["one", "two"]]

    def test_kpt_shape_valid(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with valid yaml data."""
        config: _YoloPoseDatasetConfig = _YoloPoseDatasetConfig(
            raw_yaml_data=yaml_pose_data
        )
        assert config.kpt_shape == (2, 2)

    def test_kpt_shape_too_few_elements(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with missing elements of the tuple."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2,)

        with pytest.raises(
            YoloDatasetConfigError, match="shape of the 'kpt_shape' parameter"
        ):
            _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_data)

    def test_kpt_shape_too_many_elements(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with extra elements of the tuple."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, 3, 2)

        with pytest.raises(
            YoloDatasetConfigError, match="shape of the 'kpt_shape' parameter"
        ):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_num_dimensions_not_2_or_3(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property when num_dimensions is not 2 or 3."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, 1)

        with pytest.raises(YoloDatasetConfigError, match="2 or 3, got 1."):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_non_integer_entries(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property when one or both entries is not an integer."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, "1")

        with pytest.raises(YoloDatasetConfigError, match="to be an integer, got"):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_num_keypoints_not_eq_to_length_of_kpt_names(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests the kpt_shape property when the num_keypoints entry is not equal to the length of kpt_names."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (5, 2)

        with pytest.raises(
            YoloDatasetConfigError, match="does not equal the number of keypoints"
        ):
            _ = _YoloPoseDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape
