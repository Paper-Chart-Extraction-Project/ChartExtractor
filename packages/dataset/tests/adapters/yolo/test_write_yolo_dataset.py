"""A module with functions that write a VisionDataset into a Yolo formatted dataset on disk."""

from pathlib import Path

import pytest

from dataset.adapters.yolo.yolo_dataset import _YoloDataset


class TestCreateDirectoryStructure:
    """A collection of tests for the create_directory_structure function."""

    @pytest.mark.skip(reason="Not yet implemented.")
    def test_no_eval_sets(self, yolo_detect_dataset: _YoloDataset, tmp_path: Path):
        """Tests the function with a dataset that has no optional evaluation sets."""

    @pytest.mark.skip(reason="Not yet implemented.")
    def test_with_eval_sets(self, yolo_detect_dataset: _YoloDataset, tmp_path: Path):
        """Tests the function with a dataset that has optional evaluation sets."""


class TestWriteDatasetYamltoDisk:
    """A collection of tests for the write_dataset_yaml_to_disk function."""

    @pytest.mark.skip(reason="Not yet implemented.")
    def test_write_detect_dataset_yaml_no_eval_sets(
        self, yolo_detect_dataset: _YoloDataset, tmp_path: Path
    ):
        """Tests the function with a detect dataset and no optional evaluation sets."""

    @pytest.mark.skip(reason="Not yet implemented.")
    def test_write_detect_dataset_yaml_with_eval_sets(
        self, yolo_detect_dataset: _YoloDataset, tmp_path: Path
    ):
        """Tests the function with a detect dataset and optional evaluation sets."""

    @pytest.mark.skip(reason="Not yet implemented.")
    def test_write_pose_dataset_yaml(
        self, yolo_detect_dataset: _YoloDataset, tmp_path: Path
    ):
        """Tests the function with a pose dataset."""
