"""A test module that tests the yolo_detect_dataset module."""

import pytest

from dataset.adapters.yolo import YoloDatasetConfigError
from dataset.adapters.yolo.yolo_dataset import _RawYamlData
from dataset.adapters.yolo.yolo_detect_dataset import _YoloDetectDatasetConfig


class TestYoloDetectDatasetConfig:
    """A collection of tests for the _YoloDetectDatasetConfig class."""

    def test_all_necessary_keys_in_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that no error is raised when all necessary keys exist for a detect dataset."""
        _YoloDetectDatasetConfig(raw_yaml_data=yaml_detect_data)

    def test_necessary_key_missing_from_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that an error is raised when a necessary key doesn't exist in the raw yaml file."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data.pop("train")

        with pytest.raises(
            YoloDatasetConfigError, match="don't exist in the yaml file: train"
        ):
            _YoloDetectDatasetConfig(raw_yaml_data=altered_yaml_detect_data)

    def test_all_necessary_keys_missing_from_raw_detect_yaml_data(
        self, yaml_detect_data: _RawYamlData
    ):
        """Tests that an error is raised when all the necessary keys don't exist in the raw yaml file."""
        altered_yaml_detect_data: _RawYamlData = yaml_detect_data
        altered_yaml_detect_data.pop("train")
        altered_yaml_detect_data.pop("val")
        altered_yaml_detect_data.pop("names")

        with pytest.raises(
            YoloDatasetConfigError,
            match="don't exist in the yaml file: names, train, val.",
        ):
            _YoloDetectDatasetConfig(raw_yaml_data=altered_yaml_detect_data)
