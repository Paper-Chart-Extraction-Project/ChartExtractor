"""A test module that tests the yolo_pose_dataset module."""

import textwrap
from pathlib import Path

import pytest
from core.annotations import (
    BoundingBox,
    BoundingBoxWithKeypoints,
    Keypoint,
    VisibilityStatus,
)
from core.geometry import Point, Rectangle

from dataset.adapters.yolo import (
    YoloDatasetConfigError,
    YoloDatasetError,
)
from dataset.adapters.yolo.yolo_dataset import _RawYamlData
from dataset.adapters.yolo.yolo_pose_dataset import (
    _YoloPoseDataset,
    _YoloPoseDatasetConfig,
)


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


class TestYoloPoseDatasetAnnotationReading:
    def test_read_pose_annotation_line_valid(
        self, yolo_pose_dataset_no_vis: _YoloPoseDataset
    ):
        """Tests that a valid pose annotation line can be read."""
        valid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 0.48 0.54\n"
        read_annotation: BoundingBoxWithKeypoints = (
            yolo_pose_dataset_no_vis.read_yolo_pose_annotation(valid_line)
        )
        true_bounding_box: BoundingBox = BoundingBox(
            rectangle=Rectangle.from_center_xywh(0.5, 0.55, 0.1, 0.2), category="test_0"
        )
        true_keypoints: list[Keypoint] = [
            Keypoint(
                point=Point(x=0.5, y=0.57),
                category="kpt_1",
                visibility=VisibilityStatus.VISIBLE,
            ),
            Keypoint(
                point=Point(x=0.48, y=0.54),
                category="kpt_2",
                visibility=VisibilityStatus.VISIBLE,
            ),
        ]
        true_annotation: BoundingBoxWithKeypoints = BoundingBoxWithKeypoints(
            bounding_box=true_bounding_box, keypoints=true_keypoints
        )

        assert (
            true_annotation.bounding_box.rectangle
            == read_annotation.bounding_box.rectangle
        )
        assert (
            true_annotation.bounding_box.category
            == read_annotation.bounding_box.category
        )
        for read_kpt, true_kpt in zip(
            read_annotation.keypoints, true_annotation.keypoints
        ):
            assert read_kpt.point.x == true_kpt.point.x
            assert read_kpt.point.y == true_kpt.point.y
            assert read_kpt.category == true_kpt.category

    def test_read_pose_annotation_line_too_many_entries(
        self, yolo_pose_dataset_no_vis: _YoloPoseDataset
    ):
        """Tests that an exception is raised when there are too many items in the line."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 2 0.48 0.54 2\n"

        with pytest.raises(YoloDatasetError, match="Too many entries"):
            yolo_pose_dataset_no_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_line_too_few_entries(
        self, yolo_pose_dataset_no_vis: _YoloPoseDataset
    ):
        """Tests that an exception is raised when there are too few items in the line."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57\n"

        with pytest.raises(YoloDatasetError, match="Too few entries"):
            yolo_pose_dataset_no_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_line_with_visibility_data(
        self, yolo_pose_dataset_with_vis: _YoloPoseDataset
    ):
        """Tests reading an annotation that contains visibility data."""
        valid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 0 0.48 0.54 1\n"

        read_annotation: BoundingBoxWithKeypoints = (
            yolo_pose_dataset_with_vis.read_yolo_pose_annotation(valid_line)
        )
        true_bounding_box: BoundingBox = BoundingBox(
            rectangle=Rectangle.from_center_xywh(0.5, 0.55, 0.1, 0.2), category="test_0"
        )
        true_keypoints: list[Keypoint] = [
            Keypoint(
                point=Point(x=0.5, y=0.57),
                category="kpt_1",
                visibility=VisibilityStatus.MISSING,
            ),
            Keypoint(
                point=Point(x=0.48, y=0.54),
                category="kpt_2",
                visibility=VisibilityStatus.OCCLUDED,
            ),
        ]
        true_annotation: BoundingBoxWithKeypoints = BoundingBoxWithKeypoints(
            bounding_box=true_bounding_box, keypoints=true_keypoints
        )

        assert (
            true_annotation.bounding_box.rectangle
            == read_annotation.bounding_box.rectangle
        )
        assert (
            true_annotation.bounding_box.category
            == read_annotation.bounding_box.category
        )
        for read_kpt, true_kpt in zip(
            read_annotation.keypoints, true_annotation.keypoints
        ):
            assert read_kpt.point.x == true_kpt.point.x
            assert read_kpt.point.y == true_kpt.point.y
            assert read_kpt.category == true_kpt.category
            assert read_kpt.visibility == true_kpt.visibility

    def test_read_pose_annotation_line_with_erronous_visibility_data(
        self, yolo_pose_dataset_with_vis: _YoloPoseDataset
    ):
        """Tests reading an annotation that contains visibility data that is not in [0, 1, 2]."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 2 0.48 0.54 4\n"

        with pytest.raises(YoloDatasetError, match="is not in \\[0, 1, 2\\]."):
            yolo_pose_dataset_with_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_valid_file(
        self, tmp_path: Path, yolo_pose_dataset_with_vis: _YoloPoseDataset
    ):
        """Tests reading a valid file of pose annotations from disk."""
        file_contents: str = textwrap.dedent("""
        0 0.5 0.5 0.1 0.1 0.49 0.51 2 0.46 0.54 1
        1 0.2 0.8 0.05 0.15 0.22 0.87 2 0.19 0.79 2
        0 0.03 0.88 0.03 0.11 0.029 0.888 2 0.02 0.85 0
        """).strip("\n")
        temp_detect_annotation_filepath: Path = tmp_path / "temp_detect_annotation.txt"
        temp_detect_annotation_filepath.write_text(file_contents)

        yolo_pose_dataset_with_vis.read_yolo_annotation_file(
            temp_detect_annotation_filepath
        )

    def test_read_pose_annotation_invalid_file(
        self, tmp_path: Path, yolo_pose_dataset_with_vis: _YoloPoseDataset
    ):
        """Tests that the line an error occurs on is put into the exception when it occurs."""
        file_contents: str = textwrap.dedent("""
        0 0.5 0.5 0.1 0.1 0.49 0.51 2 0.46 0.54 1
        1 0.2 0.8 0.05 0.15 0.22 0.87 2 0.19 0.79 2
        0 0.03 0.88 0.03 0.11 0.029 0.888 2 0.02 0.85
        """).strip("\n")
        temp_detect_annotation_filepath: Path = tmp_path / "temp_detect_annotation.txt"
        temp_detect_annotation_filepath.write_text(file_contents)

        with pytest.raises(YoloDatasetError, match="while reading line 3."):
            yolo_pose_dataset_with_vis.read_yolo_annotation_file(
                temp_detect_annotation_filepath
            )
