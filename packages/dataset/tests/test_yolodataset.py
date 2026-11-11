"""A test module that tests the yolo_dataset module."""

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
from dataset.adapters.yolo import YoloDatasetError, YoloDatasetWarning
from dataset.adapters.yolo.yolo_dataset import (
    _RawYamlData,
    _YoloDataset,
    _YoloDatasetConfig,
    _YoloDatasetType,
)


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

        with pytest.raises(YoloDatasetError, match="'kpt_names' but not 'kpt_shape'"):
            _YoloDatasetConfig(raw_yaml_data=altered_yaml_pose_data).dataset_type  # noqa: B018

    def test_dataset_type_pose_but_no_kpt_names_is_supplied(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests that the dataset type is set to 'pose' when a yaml file with 'kpt_names' is missing."""
        altered_yaml_pose_data: _RawYamlData = yaml_pose_data
        altered_yaml_pose_data.pop("kpt_names")

        with pytest.raises(YoloDatasetError, match="'kpt_shape' but not 'kpt_names'"):
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

    def test_names_is_a_sorted_list(self, yaml_detect_data: _RawYamlData):
        """Tests that the names list is created from the dictionary and sorts properly."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)
        assert config.names == ["test_0", "test_1", "test_2"]

    def test_kpt_names_is_a_sorted_list(self, yaml_pose_data: _RawYamlData):
        """Tests that the names list is created from the dictionary and sorts properly."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_pose_data)
        assert config.kpt_names == [["left", "right"], ["one", "two"]]

    def test_kpt_shape_valid(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with valid yaml data."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_pose_data)
        assert config.kpt_shape == (2, 2)

    def test_kpt_shape_for_detect_dataset(self, yaml_detect_data: _RawYamlData):
        """Tests the kpt_shape property is None with detect yaml data."""
        config: _YoloDatasetConfig = _YoloDatasetConfig(raw_yaml_data=yaml_detect_data)
        assert config.kpt_shape is None

    def test_kpt_shape_too_few_elements(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with missing elements of the tuple."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2,)

        with pytest.raises(
            YoloDatasetError, match="shape of the 'kpt_shape' parameter"
        ):
            _ = _YoloDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_too_many_elements(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property with extra elements of the tuple."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, 3, 2)

        with pytest.raises(
            YoloDatasetError, match="shape of the 'kpt_shape' parameter"
        ):
            _ = _YoloDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_num_dimensions_not_2_or_3(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property when num_dimensions is not 2 or 3."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, 1)

        with pytest.raises(YoloDatasetError, match="2 or 3, got 1."):
            _ = _YoloDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_non_integer_entries(self, yaml_pose_data: _RawYamlData):
        """Tests the kpt_shape property when one or both entries is not an integer."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (2, "1")

        with pytest.raises(YoloDatasetError, match="to be an integer, got"):
            _ = _YoloDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape

    def test_kpt_shape_num_keypoints_not_eq_to_length_of_kpt_names(
        self, yaml_pose_data: _RawYamlData
    ):
        """Tests the kpt_shape property when the num_keypoints entry is not equal to the length of kpt_names."""
        altered_yaml_data: _RawYamlData = yaml_pose_data
        altered_yaml_data["kpt_shape"] = (5, 2)

        with pytest.raises(
            YoloDatasetError, match="does not equal the number of keypoints"
        ):
            _ = _YoloDatasetConfig(raw_yaml_data=altered_yaml_data).kpt_shape


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

        yds: _YoloDataset = _YoloDataset.from_folderpath(
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

        yds: _YoloDataset = _YoloDataset.from_folderpath(
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

        yds: _YoloDataset = _YoloDataset.from_folderpath(
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
            _: _YoloDataset = _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )

    def test_an_images_doesnt_exist_for_an_annotation(self, tmp_path: Path):
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
            _: _YoloDataset = _YoloDataset.from_folderpath(
                tmp_path / "test_dataset_valid_folder_structure"
            )


class TestYoloDatasetDetectAnnotationReading:
    """A collection of tests for the _YoloDataset class' detect annotation reading."""

    def test_read_detect_annotation_line_valid_annotation(
        self, yolo_detect_dataset: _YoloDataset
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
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that an error is raised with a class id that is out of bounds."""
        invalid_line: str = "-1 0.55 0.44 0.17 0.05\n"
        with pytest.raises(YoloDatasetError, match="Class index is out of range"):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

        invalid_line: str = "3 0.55 0.44 0.17 0.05\n"
        with pytest.raises(YoloDatasetError, match="Class index is out of range"):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_line_left_out_of_bounds(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that a warning is emitted when a bounding box left side is out of bounds."""
        valid_line: str = "1 0.05 0.44 0.24 0.05\n"
        with pytest.warns(YoloDatasetWarning, match="left side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_top_out_of_bounds(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that a warning is emitted when a bounding box top side is out of bounds."""
        valid_line: str = "1 0.55 0.02 0.24 0.05\n"
        with pytest.warns(YoloDatasetWarning, match="top side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_right_out_of_bounds(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that a warning is emitted when a bounding box right side is out of bounds."""
        valid_line: str = "1 0.95 0.26 0.24 0.17\n"
        with pytest.warns(YoloDatasetWarning, match="right side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_bottom_out_of_bounds(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that a warning is emitted when a bounding box right side is out of bounds."""
        valid_line: str = "1 0.46 0.96 0.14 0.17\n"
        with pytest.warns(YoloDatasetWarning, match="bottom side is out of bounds"):
            yolo_detect_dataset.read_yolo_detect_annotation(valid_line)

    def test_read_detect_annotation_line_too_many_entries(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that an exception is raised when there are too many items in the line."""
        invalid_line: str = "1 0.46 0.96 0.14 0.17 0.44\n"
        with pytest.raises(
            YoloDatasetError, match="Too many entries in detect annotation line."
        ):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_line_too_few_entries(
        self, yolo_detect_dataset: _YoloDataset
    ):
        """Tests that an exception is raised when there are too few items in the line."""
        invalid_line: str = "1 0.46 0.96 0.14\n"
        with pytest.raises(
            YoloDatasetError, match="Too few entries in detect annotation line."
        ):
            yolo_detect_dataset.read_yolo_detect_annotation(invalid_line)

    def test_read_detect_annotation_valid_file(
        self, tmp_path: Path, yolo_detect_dataset: _YoloDataset
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
        self, tmp_path: Path, yolo_detect_dataset: _YoloDataset
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

    def test_read_pose_annotation_line_valid(
        self, yolo_pose_dataset_no_vis: _YoloDataset
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
        self, yolo_pose_dataset_no_vis: _YoloDataset
    ):
        """Tests that an exception is raised when there are too many items in the line."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 2 0.48 0.54 2\n"

        with pytest.raises(YoloDatasetError, match="Too many entries"):
            yolo_pose_dataset_no_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_line_too_few_entries(
        self, yolo_pose_dataset_no_vis: _YoloDataset
    ):
        """Tests that an exception is raised when there are too few items in the line."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57\n"

        with pytest.raises(YoloDatasetError, match="Too few entries"):
            yolo_pose_dataset_no_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_line_with_visibility_data(
        self, yolo_pose_dataset_with_vis: _YoloDataset
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
        self, yolo_pose_dataset_with_vis: _YoloDataset
    ):
        """Tests reading an annotation that contains visibility data that is not in [0, 1, 2]."""
        invalid_line: str = "0 0.5 0.55 0.1 0.2 0.5 0.57 2 0.48 0.54 4\n"

        with pytest.raises(YoloDatasetError, match="is not in \\[0, 1, 2\\]."):
            yolo_pose_dataset_with_vis.read_yolo_pose_annotation(invalid_line)

    def test_read_pose_annotation_valid_file(
        self, tmp_path: Path, yolo_pose_dataset_with_vis: _YoloDataset
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
        self, tmp_path: Path, yolo_pose_dataset_with_vis: _YoloDataset
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
