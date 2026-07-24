"""Contains the Image class."""

import hashlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import BaseModel, field_validator

_PIL_IMAGE_MODE_TO_CHANNELS_MAP: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "LAB": 3,
    "HSV": 3,
}


class _ImageSize(NamedTuple):
    """A named tuple containing an image's size"""

    height: int
    width: int
    channels: int


class ImageMetadata(BaseModel):
    """An Image's metadata.

    The metadata about an image. Can be loaded into an Image object with load().

    Attributes:
        path (Path | None):
            The path to the image file on disk. If an unsaved Image is unloaded, or a
            transformation is applied to an Image this ImageMetadata is attached to, path is set to
            None.
        skip_exif_transpose (bool):
            Whether or not to skip transposing the image with exif_transpose.
            Useful for if images have already been loaded using exif_transpose.
            Defaults to False.
        size (tuple[int, int, int] | None):
            The (height, width, channels) of the image. If path is None, returns None.
            Not the raw image size, but the image size *after* exif data, if applicable.
        height (int):
            The height of the image.
            Not the raw image height, but the image height *after* exif data, if applicable.
        width (int):
            The width of the image.
            Not the raw image width, but the image width *after* exif data, if applicable.
        channels (int):
            The number of channels of the image.
    """

    path: Path | None
    skip_exif_transpose: bool = False

    @field_validator("path", mode="before")
    @classmethod
    def check_path_is_valid(cls, p: Path | None) -> Path | None:
        """Checks if the path supplied (1) is not None, (2) exists, (3) is not a directory.

        If constructing an ImageMetadata object directly, the path must point to a legitimate
        image. Path can only be set to None through the Image class, such as loading an Image object
        from an existing PIL or cv2 image, or from transforming an Image object.

        Args:
            p (Path):
                The path to check.

        Returns:
            The path supplied, if it is valid.

        Raises:
            FileNotFoundError:
                If the path does not exist.
            IsADirectoryError:
                If the path leads to a directory, not a file.
        """
        if p is None:
            return p
        if not p.exists():
            raise FileNotFoundError(f"Path to image {p.resolve()} does not exist.")
        if p.is_dir():
            raise IsADirectoryError(
                f"Path to image {p.resolve()} is a directory, not a file."
            )

        return p

    @cached_property
    def size(self) -> _ImageSize | None:
        """The (width, height, channels) of the image.

        Not the raw image size, but the image size *after* exif data, if applicable.
        """
        if self.path is None:
            return None
        with PILImage.open(self.path) as pil_img:
            if not self.skip_exif_transpose:
                pil_img: PILImage.Image = ImageOps.exif_transpose(pil_img)
            size: tuple[int, int, int] = ImageMetadata._get_size_from_pil_image(pil_img)
        return size

    @property
    def height(self) -> int | None:
        """The height of the image.

        Not the raw image height, but the image height *after* exif data, if applicable.
        """
        return self.size.height if self.size is not None else None

    @property
    def width(self) -> int | None:
        """The width of the image.

        Not the raw image width, but the image width *after* exif data, if applicable.
        """
        return self.size.width if self.size is not None else None

    @property
    def channels(self) -> int | None:
        """The number of channels of the image."""
        return self.size.channels if self.size is not None else None

    @staticmethod
    def _get_size_from_pil_image(pil_img: PILImage.Image) -> _ImageSize:
        """Gets the size from a pil image.

        Args:
            pil_img (PILImage.Image):
                The PIL Image to get the size of. Does not need to be loaded into memory.

        Returns:
            The PIL Image's height, width, and channels.
        """
        width: int = pil_img.size[0]
        height: int = pil_img.size[1]
        # defaults to the getbands() method if the mode isn't recognized.
        channels: int | None = _PIL_IMAGE_MODE_TO_CHANNELS_MAP.get(
            pil_img.mode, len(pil_img.getbands())
        )

        return _ImageSize(height=height, width=width, channels=channels)

    def load(self) -> Image:
        """Loads the image from the ImageRecord's path."""
        return Image._from_metadata(self)


@dataclass(slots=True)
class Image:
    """An intermediate representation of an image.

    Used as an intermediary between PIL/cv2 and the rest of the code.

    Attributes:
        size (tuple[int, int, int]):
            The (width, height, channels) of the image.
        width (int):
            The width of the image.
        height (int):
            The height of the image.
        channels (int):
            The number of channels the image has.
    """

    _data: np.ndarray
    _metadata: ImageMetadata | None
    _has_been_edited_without_saving: bool = False

    @property
    def size(self) -> _ImageSize:
        """The (width, height, channels) of the image."""
        shape: tuple = self._data.shape
        match len(shape):
            case 2:
                height, width = self._data.shape
                channels = 1
            case 3:
                height, width, channels = self._data.shape
            case _:
                raise ValueError(
                    f"Cannot compute the size of an image with shape {self._data.shape}"
                )

        return _ImageSize(height=height, width=width, channels=channels)

    @property
    def height(self) -> int:
        """The height of the image."""
        return self.size.height

    @property
    def width(self) -> int:
        """The width of the image."""
        return self.size.width

    @property
    def channels(self) -> int:
        """The number of channels the image has."""
        return self.size.channels

    @classmethod
    def from_path(cls, path: Path) -> Image:
        """Creates an Image from a path on disk.

        Args:
            path (Path):
                The path to the image on disk.

        Returns:
            An Image from the image path supplied.
        """
        return cls._from_metadata(ImageMetadata(path=path))

    @classmethod
    def from_pil(
        cls, pil_img: PILImage.Image, skip_exif_transpose: bool = False
    ) -> Image:
        """Creates an Image from a PIL Image.

        Args:
            pil_img (PILImage.Image):
                The PIL Image to make an Image from.
            skip_exif_transpose (bool):
                Whether or not to skip transposing the image with exif_transpose.
                Useful for if images have already been loaded using exif_transpose.
                Defaults to False.

        Returns:
            An Image from the data in the PIL Image.
        """
        if not skip_exif_transpose:
            pil_img = ImageOps.exif_transpose(pil_img)
        return Image(
            _data=np.array(pil_img),
            _metadata=None,
            _has_been_edited_without_saving=True,
        )

    @classmethod
    def from_cv2(cls, cv2_image: np.ndarray, convert_bgr_to_rgb: bool = True) -> Image:
        """Creates an Image from a cv2 image array.

        Args:
            cv2_image (np.ndarray):
                The cv2 image to make an Image from.
            convert_bgr_to_rgb (bool):
                Whether or not the cv2_image is in BGR instead of RGB. If true, converts from BGR
                to RGB.
                Defaults to True.

        Returns:
            An Image with the cv2_image's data.
        """
        if convert_bgr_to_rgb:
            if cv2_image.ndim == 3 and cv2_image.shape[2] == 4:
                im_data = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
            elif cv2_image.ndim == 3 and cv2_image.shape[2] == 3:
                im_data = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            else:
                num_channels: int = (
                    1 if len(cv2_image.shape) == 2 else cv2_image.shape[2]
                )
                raise ValueError(
                    "convert_bgr_to_rgb set to True in Image.from_cv2, but number of channels "
                    + f"is neither 3 nor 4 (num_channels={num_channels})"
                )
        else:
            im_data = cv2_image

        return Image(
            _data=im_data, _metadata=None, _has_been_edited_without_saving=True
        )

    def transform(
        self, transformation_function: Callable[..., np.ndarray], **kwargs
    ) -> None:
        """Applies a transformation function to the image.

        Args:
            transformation_function (Callable[..., np.ndarray]):
                The function used to transform the image. Must take a numpy ndarray as the image
                data, and return an ndarray as output.
            **kwargs:
                Additional keyword arguments used in the transformation function.

        Raises:
            ValueError:
                If the output of the the transformation function is not an ndarray.
        """
        result: Any = transformation_function(self._data, **kwargs)
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Transformation function must return np.ndarray, got {type(result)}"
            )
        self._data = result
        self._metadata = None
        self._has_been_edited_without_saving = True

    def save(self, output_path: Path, force_overwrite: bool = False) -> None:
        """Saves the image to disk.

        Args:
            output_path (Path):
                The location to save the image.
            force_overwrite (bool):
                Whether or not to forcably overwrite the image data if an image already exists.
                Defaults to False.

        Raises:
            IsADirectoryError:
                If the location supplied to the output_path is a directory and not a file.
            ValueError:
                If force_overwrite is False, and a file exists at output_path.
        """
        if output_path.is_dir():
            raise IsADirectoryError(
                f"The output path ({output_path.resolve()}) to save the image to is a directory, "
                + "not a file."
            )
        if not force_overwrite and output_path.exists():
            raise ValueError(
                f"A file already exists at the output path ({output_path.resolve()}).\n"
                + "To overwrite the file, set force_overwrite=True."
            )

        self.to_pil().save(output_path)
        self._metadata = ImageMetadata(path=output_path)
        self._has_been_edited_without_saving = False

    def sha256_hash(self) -> str:
        """The SHA256 hash of the image's data.

        The hash is as deterministic as possible by (1) excluding image metadata and (2) enforcing
        a contiguous array of image data in memory.

        Returns:
            The SHA256 hash of the image's data.
        """
        self._data = np.ascontiguousarray(self._data)
        return hashlib.sha256(self._data.tobytes()).hexdigest()

    def unload(self, error_if_not_saved: bool = True) -> ImageMetadata | None:
        """Removes image data from memory and returns the image's metadata.

        Args:
            error_if_not_saved (bool):
                Raises an error if attempting to unload without first saving.

        Returns:
            Either an ImageMetadata object with the Image's metadata, or None, if the image had
            not been saved first.
        """
        unload_without_saving_msg: str = (
            "Image is being unloaded without having been saved."
        )
        if self._has_been_edited_without_saving:
            if error_if_not_saved:
                raise ValueError(unload_without_saving_msg)
            warnings.warn(unload_without_saving_msg, UserWarning)

        metadata: ImageMetadata = self._metadata
        self._metadata = None
        self._data = None

        if self._has_been_edited_without_saving:
            return None

        self.__class__ = UnloadedImage

        return metadata

    def to_ndarray(self) -> np.ndarray:
        """Returns *a copy* of the underlying image data."""
        return self._data.copy()

    def to_pil(self) -> PILImage.Image:
        """Returns the Image as a PIL Image."""
        return PILImage.fromarray(self._data)

    def to_cv2(self) -> np.ndarray:
        """Returns the Image as a cv2 image

        Essentially the same data as BGR instead of RGB.
        """
        data: np.ndarray = self.to_ndarray()
        if self.channels == 3:
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if self.channels == 4:
            return cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
        return data  # if its not 3 or 4 channels, then its a greyscale image.

    @classmethod
    def _from_metadata(cls, metadata: ImageMetadata) -> Image:
        """Creates an Image from an ImageMetadata class.

        Only used by ImageMetadata. Identical to calling ImageMetadata.load().

        Args:
            metadata (ImageMetadata):
                The path to the image on disk, along with other metadata.

        Returns:
            The image from disk loaded into memory as an Image object.
        """
        match metadata.channels:
            case 1:
                mode_to_convert_to: str = "L"
            case 3:
                mode_to_convert_to: str = "RGB"
            case 4:
                mode_to_convert_to: str = "RGBA"
            case _:
                raise ValueError(
                    f"Cannot load image with {metadata.channels} channels."
                )
        with PILImage.open(metadata.path) as pil_img:
            if not metadata.skip_exif_transpose:
                pil_img = ImageOps.exif_transpose(pil_img)
            pil_img: PILImage.Image = pil_img.convert(mode_to_convert_to)
            image_data: np.ndarray = np.array(pil_img)

        return Image(_data=image_data, _metadata=metadata)


@dataclass(slots=True)
class UnloadedImage:
    """Represents the state of an unloaded image.

    This class exists to prevent users mistakenly trying to work with images after calling the
    unload() method. This class raises an error when attempting anything.
    """

    _data: None = None
    _metadata: None = None
    _has_been_edited_without_saving: None = None

    def __getattribute__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        raise RuntimeError("Process Error: This image has already been unloaded.")
