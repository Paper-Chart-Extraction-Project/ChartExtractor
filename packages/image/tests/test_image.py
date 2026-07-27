"""A series of tests for the Image class."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import TEST_DATA_DIR
import os
from PIL import Image as PILImage
import sys

from image import Image, ImageMetadata, UnloadedImage


class TestSize:
    """A group of tests for the size property."""

    def test_three_by_three_rgb_image(self, three_by_three_rgb_test_path: Path):
        """Tests the size property with an image that is 3x3 rgb."""
        im: Image = Image.from_path(path=three_by_three_rgb_test_path)

        assert im.size == (3, 3, 3)

    def test_four_by_three_black_image(self, four_by_three_black_test_path: Path):
        """Tests the size property with a 4x3 image with no exif data."""
        im: Image = Image.from_path(path=four_by_three_black_test_path)

        assert im.size == (3, 4, 3)

    def test_four_by_three_exif_rotated_black_image(
        self, four_by_three_exif_rotated_black_test_path: Path
    ):
        """Tests the size property with a 4x3 image with no exif data."""
        im: Image = Image.from_path(path=four_by_three_exif_rotated_black_test_path)

        assert im.size == (4, 3, 3)

    def test_three_by_three_greyscale_image(
        self, three_by_three_greyscale_test_path: Path
    ):
        """Tests the size property with a single channel image."""
        im: Image = Image.from_path(path=three_by_three_greyscale_test_path)

        assert im.size == (3, 3, 1)


class TestConstructors:
    """A group of tests for the constructors of Image."""

    def test_from_path(self, three_by_three_rgb_test_path: Path):
        """Tests if the image can be loaded from a path."""
        im: Image = Image.from_path(path=three_by_three_rgb_test_path)

        assert im._data is not None

    def test_from_metadata_with_exif(
        self, four_by_three_exif_rotated_black_test_path: Path
    ):
        """Tests if the image can be loaded with application of exif transformations."""
        im: Image = Image.from_path(path=four_by_three_exif_rotated_black_test_path)

        assert im.size == (4, 3, 3)

    def test_from_metadata_skip_exif(
        self, four_by_three_exif_rotated_black_test_path: Path
    ):
        """Tests if the image can be loaded without applying exif transformations."""
        im_metadata: ImageMetadata = ImageMetadata(
            path=four_by_three_exif_rotated_black_test_path, skip_exif_transpose=True
        )
        im: Image = Image._from_metadata(im_metadata)

        assert im.size == (3, 4, 3)

    def test_from_pil_with_exif(self, four_by_three_exif_rotated_black_test_path: Path):
        """Tests if the image can be loaded from a PIL Image with application of exif transformations."""
        pil_img: PILImage.Image = PILImage.open(
            four_by_three_exif_rotated_black_test_path
        )
        im: Image = Image.from_pil(pil_img)

        assert im.size == (4, 3, 3)

    def test_from_pil_skip_exif(self, four_by_three_exif_rotated_black_test_path: Path):
        """Tests if the image can be loaded from a PIL Image without applying exif transformations."""
        pil_img: PILImage.Image = PILImage.open(
            four_by_three_exif_rotated_black_test_path
        )
        im: Image = Image.from_pil(pil_img, skip_exif_transpose=True)

        assert im.size == (3, 4, 3)

    def test_from_cv2_convert_bgr_to_rgb_with_5_channels(
        self, four_by_three_black_test_path: Path
    ):
        """Tests if the from_cv2 method errors when an image attempting to convert rgb to bgr has the wrong number of channels."""
        cv2_image: np.ndarray = cv2.imread(four_by_three_black_test_path)
        h, w = cv2_image.shape[:2]
        blank_channel: np.ndarray = np.zeros((h, w), dtype=cv2_image.dtype)
        cv2_image = np.dstack((cv2_image, blank_channel, blank_channel))

        with pytest.raises(ValueError, match="neither 3 nor 4"):
            Image.from_cv2(cv2_image)

    def test_from_cv2_convert_bgr_to_rgb_with_1_channel(
        self, three_by_three_greyscale_test_path: Path
    ):
        """Tests if the from_cv2 method errors when an image attempting to convert rgb to bgr has the wrong number of channels."""
        cv2_image: np.ndarray = cv2.imread(three_by_three_greyscale_test_path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

        with pytest.raises(ValueError, match="neither 3 nor 4"):
            Image.from_cv2(cv2_image)

    def test_from_pil_and_from_cv2_equivalent(self, three_by_three_rgb_test_path: Path):
        """Tests if the from_pil and from_cv2 produce equivalent sha256 hashes."""
        pil_image: PILImage.Image = PILImage.open(three_by_three_rgb_test_path)
        cv2_image: np.ndarray = cv2.imread(three_by_three_rgb_test_path)

        image_from_pil: Image = Image.from_pil(pil_image)
        image_from_cv2: Image = Image.from_cv2(cv2_image)

        assert image_from_pil.sha256_hash() == image_from_cv2.sha256_hash()

    def test_from_cv2_already_rgb(self, three_by_three_rgb_test_path: Path):
        """Tests if the from_cv2 constructor can pass through an already rgb image."""
        bgr_cv2_img: np.ndarray = cv2.imread(three_by_three_rgb_test_path)
        rgb_cv2_img = cv2.cvtColor(bgr_cv2_img, cv2.COLOR_BGR2RGB)

        bgr_image: Image = Image.from_cv2(bgr_cv2_img)
        rgb_image: Image = Image.from_cv2(rgb_cv2_img, convert_bgr_to_rgb=False)

        assert bgr_image.sha256_hash() == rgb_image.sha256_hash()


class TestRoundTrips:
    """Tests the to_pil/from_pil & to_cv2/from_cv2 methods by 'round tripping.'

    Round tripping is calling to_x, then loading a new Image from_x and checking if the objects
    are identical.
    Image metadata will be affected, so comparing it will be skipped.
    """

    def test_pil_round_trip(self, three_by_three_rgb_test_path: Path):
        """Checks that an image is equal to its pil round tripped counterpart."""
        original_image: Image = Image.from_path(three_by_three_rgb_test_path)
        round_tripped_image: Image = Image.from_pil(original_image.to_pil())

        assert original_image.sha256_hash() == round_tripped_image.sha256_hash()

    def test_cv2_round_trip(self, three_by_three_rgb_test_path: Path):
        """Checks that an image is equal to its cv2 round tripped counterpart."""
        original_image: Image = Image.from_path(three_by_three_rgb_test_path)
        round_tripped_image: Image = Image.from_cv2(original_image.to_cv2())

        assert original_image.sha256_hash() == round_tripped_image.sha256_hash()


class TestSha256Hash:
    """A group of tests for the sha256_hash method."""

    def test_all_test_image_hashes_different(
        self,
        three_by_three_rgb_test_path: Path,
        three_by_three_greyscale_test_path: Path,
        four_by_three_black_test_path: Path,
    ):
        """A sanity-check test to ensure that the sha256_hash method is hashing something."""
        # does not include the 4x3 black exif rotated image because it hashes to the same val by design
        hashes: list[str] = [
            Image.from_path(three_by_three_rgb_test_path).sha256_hash(),
            Image.from_path(three_by_three_greyscale_test_path).sha256_hash(),
            Image.from_path(four_by_three_black_test_path).sha256_hash(),
        ]
        all_hashes_are_unique: bool = len(hashes) == len(set(hashes))

        assert all_hashes_are_unique

    def test_exif_rotated_image_hashes_to_the_same_value(
        self,
        four_by_three_black_test_path: Path,
        four_by_three_exif_rotated_black_test_path: Path,
    ):
        """tests if an image with exif rotation hashes to the same value as the non-rotated version."""
        # does not include the 4x3 black exif rotated image because it hashes to the same val by design
        non_exif_hash: str = Image.from_path(
            four_by_three_black_test_path
        ).sha256_hash()
        exif_hash: str = Image.from_path(
            four_by_three_exif_rotated_black_test_path
        ).sha256_hash()

        assert non_exif_hash == exif_hash


class TestTransform:
    """A group of tests for the transform method."""

    def test_transform_erases_metadata(self, three_by_three_rgb_test_path: Path):
        """Tests that the transform method erases metadata."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        transformation_func = lambda data: data
        im.transform(transformation_func)

        assert im._metadata is None

    def test_transform_accepts_kwargs(self, three_by_three_rgb_test_path: Path):
        """Tests that the transform method accepts and uses key word arguments."""
        original_im: Image = Image.from_path(three_by_three_rgb_test_path)
        edited_im: Image = Image.from_path(three_by_three_rgb_test_path)
        transformation_func = lambda data, amt_to_subtract: np.clip(
            data - amt_to_subtract, a_min=0.0, a_max=255.0
        ).astype(np.uint8)
        edited_im.transform(transformation_func, amt_to_subtract=42)

        assert original_im.sha256_hash() == original_im.sha256_hash()
        assert edited_im.sha256_hash() == edited_im.sha256_hash()
        assert original_im.sha256_hash() != edited_im.sha256_hash()

        edited_im: Image = Image.from_path(three_by_three_rgb_test_path)
        transformation_func = lambda data, amt_to_subtract: np.clip(
            data - amt_to_subtract, a_min=0.0, a_max=255.0
        ).astype(np.uint8)
        edited_im.transform(transformation_func, amt_to_subtract=0.0)

        assert original_im.sha256_hash() == edited_im.sha256_hash()

    def test_transform_edits_image_in_place(self, three_by_three_rgb_test_path: Path):
        """Tests that the transform method edits the image data.

        This test method doesn't check that the transform does so *correctly*, just that it does.
        """
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        original_im_hash: str = im.sha256_hash()
        transformation_func = lambda data: np.clip(data, a_min=0.0, a_max=100.0)
        im.transform(transformation_func)
        edited_im_hash: str = im.sha256_hash()

        assert original_im_hash != edited_im_hash

    def test_transform_sets_has_been_edited_without_saving(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that the _has_been_edited_without_saving flag gets set to True."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        transformation_func = lambda data: data

        assert not im._has_been_edited_without_saving

        im.transform(transformation_func)

        assert im._has_been_edited_without_saving

    def test_typeerror_when_transform_does_not_return_ndarray(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that a TypeError occurs when a transformation function doesn't return an ndarray."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        transformation_func = lambda _: "Not an ndarray!"
        with pytest.raises(TypeError, match="must return np.ndarray"):
            im.transform(transformation_func)


class TestSave:
    """A group of tests for the save method."""

    file_save_path: Path = TEST_DATA_DIR / "test_saved_image.png"

    @pytest.fixture(scope="function", autouse=True)
    @classmethod
    def teardown(cls):
        """Automatically deletes the saved file if it exists."""
        yield  # test runs here.

        if cls.file_save_path.exists():
            os.remove(cls.file_save_path)

    def test_is_a_directory_error_when_given_a_dir_output(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that an IsADirectoryError occurs when save is given a directory to save to."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        with pytest.raises(IsADirectoryError, match="is a directory, not a file."):
            im.save(TEST_DATA_DIR)

    def test_value_error_when_trying_to_overwrite_existing_file(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that a ValueError occurs when save is given an existing file and not told to overwrite it."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        im.save(self.file_save_path)
        with pytest.raises(ValueError, match="already exists"):
            im.save(self.file_save_path)

    def test_overwrite_existing_file(self, three_by_three_rgb_test_path: Path):
        """Tests that a ValueError occurs when save is given an existing file and not told to overwrite it."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        im.save(self.file_save_path)
        im.save(self.file_save_path, force_overwrite=True)

    def test_metadata_set_to_output_image_path(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that the image metadata is set to the new location when saving."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        original_metadata: ImageMetadata = im._metadata
        im.save(self.file_save_path)
        new_metadata: ImageMetadata = im._metadata

        assert original_metadata != new_metadata

    def test_has_been_edited_without_saving_reset_to_false(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that the _has_been_edited_without_saving flag is reset to False when saving."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        im.transform(lambda data: data)
        im.save(self.file_save_path)

        assert not im._has_been_edited_without_saving


class TestUnload:
    """A group of tests for the unload method."""

    def test_image_is_transformed_into_unloaded_image(
        self, three_by_three_rgb_test_path: Path
    ):
        """Tests that the image's values are set to None."""
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        im.unload()

        assert isinstance(im, UnloadedImage)
        with pytest.raises(RuntimeError, match="already been unloaded"):
            _ = im._data

    def test_image_array_is_freed(self, three_by_three_rgb_test_path: Path):
        """Verifies that unloading the image actually cuts its connection to the numpy array.

        Uses reference counting to ensure that the image array is disconnected from the Image
        object.
        """
        # load the image. The _data numpy array has 1 reference held by im._data.
        im: Image = Image.from_path(three_by_three_rgb_test_path)
        # Get a pointer to the array (increases the reference count to 2)
        underlying_arr: np.ndarray = im._data

        # Capture the number of references before unloading (should be 2)
        number_of_refs_before_unloading: int = sys.getrefcount(underlying_arr)

        # Unload the image. Should reduce the number of references to 1.
        im.unload()

        # Capture the number of references after unloading (should be 1).
        number_of_refs_after_unloading: int = sys.getrefcount(underlying_arr)

        # Assert that the number of references after unloading is one fewer than before.
        assert number_of_refs_after_unloading == number_of_refs_before_unloading - 1

    def test_metadata_is_retained(self, three_by_three_rgb_test_path: Path):
        """Tests that the metadata persists after object hollowing."""
        metadata: ImageMetadata = ImageMetadata(path=three_by_three_rgb_test_path)
        im: Image = metadata.load()
        retained_metadata: ImageMetadata = im.unload()

        assert metadata == retained_metadata
