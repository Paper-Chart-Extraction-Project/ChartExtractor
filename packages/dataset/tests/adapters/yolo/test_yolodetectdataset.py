"""A test module that tests the yolo_detect_dataset module."""

import textwrap
from pathlib import Path

import pytest
from core.annotations import BoundingBox
from core.geometry import Rectangle

from dataset import DetectVisionDataset
from dataset.adapters.yolo import (
    YoloDatasetConfigError,
    YoloDatasetError,
    YoloDatasetWarning,
)
from dataset.adapters.yolo.yolo_dataset import _RawYamlData
from dataset.adapters.yolo.yolo_detect_dataset import (
    _YoloDetectDataset,
    _YoloDetectDatasetConfig,
)


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


class TestYoloDetectDatasetAnnotationReading:
    """A collection of tests for the _YoloDetectDataset class' detect annotation reading."""

    def test_read_detect_annotation_line_valid_annotation(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests detect annotation reading with a valid line."""
        valid_line: str = "1 0.55 0.44 0.17 0.05\n"
        true_annotation: BoundingBox = BoundingBox(
            rectangle=Rectangle.from_center_xywh(0.55, 0.44, 0.17, 0.05),
            category="test_1",
        )
        read_annotation: BoundingBox = yolo_detect_dataset.read_yolo_detect_annotation(
            valid_line
        )
        assert true_annotation.rectangle == read_annotation.rectangle
        assert true_annotation.category == read_annotation.category

    def test_read_detect_annotation_line_class_id_out_of_bounds(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that an error is raised with a class id that is out of bounds."""
        invalid_line: str = "-1 0.55 0.44 0.17 0.05\n"
        with pytest.raises(YoloDatasetError, match="Class index is out of range"):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

        invalid_line: str = "3 0.55 0.44 0.17 0.05\n"
        with pytest.raises(YoloDatasetError, match="Class index is out of range"):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_line_left_out_of_bounds(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that a warning is emitted when a bounding box left side is out of bounds."""
        valid_line: str = "1 0.05 0.44 0.24 0.05\n"
        with pytest.warns(YoloDatasetWarning, match="left side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_top_out_of_bounds(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that a warning is emitted when a bounding box top side is out of bounds."""
        valid_line: str = "1 0.55 0.02 0.24 0.05\n"
        with pytest.warns(YoloDatasetWarning, match="top side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_right_out_of_bounds(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that a warning is emitted when a bounding box right side is out of bounds."""
        valid_line: str = "1 0.95 0.26 0.24 0.17\n"
        with pytest.warns(YoloDatasetWarning, match="right side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_bottom_out_of_bounds(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that a warning is emitted when a bounding box right side is out of bounds."""
        valid_line: str = "1 0.46 0.96 0.14 0.17\n"
        with pytest.warns(YoloDatasetWarning, match="bottom side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_too_many_entries(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that an exception is raised when there are too many items in the line."""
        invalid_line: str = "1 0.46 0.96 0.14 0.17 0.44\n"
        with pytest.raises(
            YoloDatasetError, match="Too many entries in detect annotation line."
        ):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_line_too_few_entries(
        self, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that an exception is raised when there are too few items in the line."""
        invalid_line: str = "1 0.46 0.96 0.14\n"
        with pytest.raises(
            YoloDatasetError, match="Too few entries in detect annotation line."
        ):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_valid_file(
        self, tmp_path: Path, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests reading a valid file of annotations from disk."""
        file_contents: str = textwrap.dedent("""
        0 0.5 0.5 0.1 0.1
        1 0.2 0.8 0.05 0.15
        0 0.03 0.88 0.03 0.11
        """).strip("\n")
        temp_detect_annotation_filepath: Path = tmp_path / "temp_detect_annotation.txt"
        temp_detect_annotation_filepath.write_text(file_contents)

        yolo_detect_dataset.read_yolo_annotation_file(temp_detect_annotation_filepath)

    def test_read_detect_annotation_invalid_file(
        self, tmp_path: Path, yolo_detect_dataset: _YoloDetectDataset
    ):
        """Tests that the line an error occurs on is put into the exception when it occurs."""
        file_contents: str = textwrap.dedent("""
        0 0.5 0.5 0.1 0.1
        1 0.2 0.8 0.05
        0 0.03 0.88 0.03 0.11
        """).strip("\n")
        temp_detect_annotation_filepath: Path = tmp_path / "temp_detect_annotation.txt"
        temp_detect_annotation_filepath.write_text(file_contents)

        with pytest.raises(YoloDatasetError, match="while reading line 2"):
            yolo_detect_dataset.read_yolo_annotation_file(
                temp_detect_annotation_filepath
            )


class TestCreateDatasetYaml:
    """A collection of tests for the _YoloDetectDatasetConfig's method for creating dataset.yaml data."""

    def test_folderpath_leads_to_file(
        self,
        tmp_path: Path,
        yaml_detect_data: _RawYamlData,
        detect_vision_dataset: DetectVisionDataset,
    ):
        """Tests that the method errors when a filepath is supplied instead of a folderpath."""
        (tmp_path / "file.txt").touch()
        ydc: _YoloDetectDatasetConfig = _YoloDetectDatasetConfig(
            raw_yaml_data=yaml_detect_data
        )
        with pytest.raises(YoloDatasetConfigError, match="file, not a folder"):
            ydc.create_dataset_yaml_for_detect_vision_dataset(
                detect_vision_dataset, tmp_path / "file.txt"
            )

    def test_folderpath_leads_to_folder(
        self,
        tmp_path: Path,
        yaml_detect_data: _RawYamlData,
        detect_vision_dataset: DetectVisionDataset,
    ):
        """Tests that the method errors when a folderpath is correctly supplied."""
        ydc: _YoloDetectDatasetConfig = _YoloDetectDatasetConfig(
            raw_yaml_data=yaml_detect_data
        )
        ydc.create_dataset_yaml_for_detect_vision_dataset(
            detect_vision_dataset, tmp_path
        )

    def test_yaml_data(
        self,
        tmp_path: Path,
        yaml_detect_data: _RawYamlData,
        detect_vision_dataset: DetectVisionDataset,
    ):
        """Tests that the dataset yaml data outputted is accurate."""
        ydc: _YoloDetectDatasetConfig = _YoloDetectDatasetConfig(
            raw_yaml_data=yaml_detect_data
        )
        computed_yaml_data: str = ydc.create_dataset_yaml_for_detect_vision_dataset(
            detect_vision_dataset, tmp_path
        )
        true_yaml_data: str = textwrap.dedent(
            f"""
            path: {tmp_path}
            train: images/train
            val: images/val

            names:
              0: a
              1: b
            """
        ).strip("\n")
        assert computed_yaml_data == true_yaml_data
