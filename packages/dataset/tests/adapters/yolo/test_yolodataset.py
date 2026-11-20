"""A test module that tests the yolo_dataset module.

For some tests that are testing the validation of the parent classes, the _YoloDetectDataset
is used, but all the code being tested is general to all child classes of _YoloDataset.
"""

import textwrap
from pathlib import Path

import pytest

from dataset.adapters.yolo import (
    YoloDatasetError,
    YoloDatasetWarning,
)
from dataset.adapters.yolo.yolo_dataset import (
    _RawYamlData,
    _YoloDataset,
    _YoloDatasetConfig,
)
from dataset.adapters.yolo.yolo_detect_dataset import (
    _YoloDetectDataset,
)


class TestYoloDatasetConfig:
    """A collection of tests for the _YoloDatasetConfig class."""

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

    def test_labels_path_constructs_properly(self, yaml_pose_data: _RawYamlData):
        """Tests that the labels path is constructed by just swapping the 'images' folder for 'labels'."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_pose_data)
        base_path: Path | None = config._raw_paths.get("base")
        assert base_path is not None
        assert config.train_labels_path == base_path / "labels" / "train"

    def test_extra_validation_set_paths(self, yaml_detect_data: _RawYamlData):
        """Tests that extra validation set paths get added to the 'validation_set_image_paths' attribute."""
        altered_yaml_pose_data: _RawYamlData = yaml_detect_data
        altered_yaml_pose_data["test_2"] = "images/test_2"
        altered_yaml_pose_data.pop("path")
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)

        assert config.validation_set_image_paths == {"test_2": Path("images/test_2")}
        assert config.validation_set_label_paths == {"test_2": Path("labels/test_2")}

    def test_names_is_a_sorted_list(self, yaml_detect_data: _RawYamlData):
        """Tests that the names list is created from the dictionary and sorts properly."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)
        assert config.names == ["test_0", "test_1", "test_2"]


class TestYoloDatasetFolderStructure:
    """A collection of tests for the _YoloDataset class' folder structure validation."""

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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
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
            _YoloDetectDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )


class TestYoloDatasetImageAndAnnotationFinding:
    """A collection of tests for the _YoloDataset class' images and annotations properties."""

    def test_no_images_or_annotations(self, tmp_path: Path):
        """Tests the properties when there are no files."""
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

        yds: _YoloDetectDataset = _YoloDetectDataset.from_folderpath(
            tmp_path / "test_dataset_valid_folder_structure"
        )

        assert yds.images == {"train": [], "val": [], "test": []}
        assert yds.annotations == {"train": [], "val": [], "test": []}

    def test_images_and_annotations(self, tmp_path: Path):
        """Tests the properties when there are regular files in the appropriate structure."""
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
        im_1_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "train"
            / "im_1.png"
        )
        im_1_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_1.txt"
        )
        im_2_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "train"
            / "im_2.jpg"
        )
        im_2_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_2.txt"
        )
        im_3_image_path: Path = (
            Path("test_dataset_valid_folder_structure") / "images" / "val" / "im_3.jpeg"
        )
        im_3_label_path: Path = (
            Path("test_dataset_valid_folder_structure") / "labels" / "val" / "im_3.txt"
        )
        files_to_create: list[Path] = [
            im_1_image_path,
            im_1_label_path,
            im_2_image_path,
            im_2_label_path,
            im_3_image_path,
            im_3_label_path,
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
        for dir in dirs_to_create + files_to_create:
            (tmp_path / dir).mkdir()
        dataset_yaml: Path = (
            tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
        )
        dataset_yaml.write_text(yaml_data)

        yds: _YoloDetectDataset = _YoloDetectDataset.from_folderpath(
            tmp_path / "test_dataset_valid_folder_structure"
        )

        assert yds.images == {
            "train": [tmp_path / im_1_image_path, tmp_path / im_2_image_path],
            "val": [tmp_path / im_3_image_path],
            "test": [],
        }
        assert yds.annotations == {
            "train": [tmp_path / im_1_label_path, tmp_path / im_2_label_path],
            "val": [tmp_path / im_3_label_path],
            "test": [],
        }

    def test_extra_validation_set(self, tmp_path: Path):
        """Tests the properties when there is an extra validation set."""
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
            Path("test_dataset_valid_folder_structure") / "labels" / "test2",
        ]
        im_1_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "test2"
            / "im_1.png"
        )
        im_1_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "test2"
            / "im_1.txt"
        )
        files_to_create: list[Path] = [
            im_1_image_path,
            im_1_label_path,
        ]
        yaml_data: str = textwrap.dedent(f"""
            path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
            train: images/train
            val: images/val
            test: images/test
            test2: images/test2

            names:
                0: test_0
                1: test_1       
        """).strip()
        for dir in dirs_to_create + files_to_create:
            (tmp_path / dir).mkdir()
        dataset_yaml: Path = (
            tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
        )
        dataset_yaml.write_text(yaml_data)

        yds: _YoloDetectDataset = _YoloDetectDataset.from_folderpath(
            tmp_path / "test_dataset_valid_folder_structure"
        )

        assert yds.images == {
            "train": [],
            "val": [],
            "test": [],
            "test2": [tmp_path / im_1_image_path],
        }
        assert yds.annotations == {
            "train": [],
            "val": [],
            "test": [],
            "test2": [tmp_path / im_1_label_path],
        }

    def test_background_images_exist(self, tmp_path: Path):
        """Tests validation when all the annotations have a corresponding image, and there are extra background images."""
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
            Path("test_dataset_valid_folder_structure") / "images" / "backgrounds",
            Path("test_dataset_valid_folder_structure") / "labels" / "backgrounds",
        ]
        background_1_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "backgrounds"
            / "background_1.png"
        )
        files_to_create: list[Path] = [
            background_1_image_path,
        ]
        yaml_data: str = textwrap.dedent(f"""
            path: {(tmp_path / "test_dataset_valid_folder_structure").resolve()}
            train: images/train
            val: images/val
            test: images/test
            backgrounds: images/backgrounds

            names:
                0: test_0
                1: test_1       
        """).strip()
        for dir in dirs_to_create + files_to_create:
            (tmp_path / dir).mkdir()
        dataset_yaml: Path = (
            tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
        )
        dataset_yaml.write_text(yaml_data)

        _YoloDataset.from_folderpath(tmp_path / "test_dataset_valid_folder_structure")

    def test_no_images_exist(self, tmp_path: Path):
        """Tests validation when no images exist for any annotation."""
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
        im_1_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_1.txt"
        )

        im_2_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_2.txt"
        )
        im_3_label_path: Path = (
            Path("test_dataset_valid_folder_structure") / "labels" / "val" / "im_3.txt"
        )
        files_to_create: list[Path] = [
            im_1_label_path,
            im_2_label_path,
            im_3_label_path,
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
        for dir in dirs_to_create + files_to_create:
            (tmp_path / dir).mkdir()
        dataset_yaml: Path = (
            tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
        )
        dataset_yaml.write_text(yaml_data)

        with pytest.raises(YoloDatasetError, match="without a corresponding image"):
            _YoloDetectDataset.from_folderpath(
                folderpath=tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_an_image_doesnt_exist_for_an_annotation(self, tmp_path: Path):
        """Tests validation when one annotation does not have a corresponding image."""
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
        im_1_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "train"
            / "im_1.png"
        )
        im_1_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_1.txt"
        )
        im_2_image_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "images"
            / "train"
            / "im_2.jpg"
        )
        im_2_label_path: Path = (
            Path("test_dataset_valid_folder_structure")
            / "labels"
            / "train"
            / "im_2.txt"
        )
        im_3_label_path: Path = (
            Path("test_dataset_valid_folder_structure") / "labels" / "val" / "im_3.txt"
        )
        files_to_create: list[Path] = [
            im_1_image_path,
            im_1_label_path,
            im_2_image_path,
            im_2_label_path,
            im_3_label_path,
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
        for dir in dirs_to_create + files_to_create:
            (tmp_path / dir).mkdir()
        dataset_yaml: Path = (
            tmp_path / "test_dataset_valid_folder_structure" / "dataset.yaml"
        )
        dataset_yaml.write_text(yaml_data)

        with pytest.raises(YoloDatasetError, match="without a corresponding image"):
            _YoloDetectDataset.from_folderpath(
                folderpath=tmp_path / "test_dataset_valid_folder_structure"
            )
