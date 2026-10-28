"""Contains the VisionDataset class."""

import warnings
from collections import Counter
from functools import reduce
from typing import Any, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .annotated_image import AnnotatedImage
from .exceptions import DatasetError, OverlappingSplitsError
from .types import ImageId
from .warnings import EmptyDatasetWarning, EmptySplitWarning


class _Splits(TypedDict):
    """A typed dictionary requiring a 'train' and 'val' key, and optional 'test' key."""

    train: Required[set[ImageId]]
    val: Required[set[ImageId]]
    test: NotRequired[set[ImageId]]


class VisionDataset(BaseModel):
    """A computer vision dataset that pairs images with annotations.

    Attributes:
        annotated_images (list[AnnotatedImage]):
            The images with annotations that form the dataset.
        dataset_metadata (dict[str, Any]):
            The datasets metadata, if it has any.
        splits (Splits):
            The splits for the dataset. Must include splits for 'train' and 'val'. May optionally
            include a 'test' split.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    annotated_images: list[AnnotatedImage] = Field(default_factory=list)
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    splits: _Splits

    @field_validator("annotated_images")
    @classmethod
    def check_all_image_ids_are_unique(
        cls, annotated_images: list[AnnotatedImage]
    ) -> list[AnnotatedImage]:
        """Ensures that all the image ids are unique."""
        all_image_ids: list[ImageId] = [im.image_id for im in annotated_images]
        if len(all_image_ids) == len(set(all_image_ids)):
            return annotated_images

        duplicated_image_ids: dict[str, int] = {
            image_id: number_of_duplicates
            for (image_id, number_of_duplicates) in Counter(all_image_ids).items()
            if number_of_duplicates > 1
        }

        err_msg: str = "Multiple images with the same image id found.\n"
        for image_id, number_of_duplicates in duplicated_image_ids.items():
            err_msg += f"{image_id} appears a total of {number_of_duplicates} times.\n"

        raise DatasetError(err_msg)

    @field_validator("splits")
    @classmethod
    def check_splits_are_disjoint(cls, splits: _Splits) -> _Splits:
        """Ensures that splits don't share images."""
        if "test" in splits:
            pairs_to_check: list[tuple[str, str]] = [
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            ]
        else:
            pairs_to_check: list[tuple[str, str]] = [("train", "val")]

        all_splits_disjoint: bool = all(
            splits[split_1].isdisjoint(splits[split_2])
            for (split_1, split_2) in pairs_to_check
        )

        if all_splits_disjoint:
            return splits

        overlap: dict[tuple[str, str], set[ImageId]] = {
            (split_1, split_2): splits[split_1].intersection(splits[split_2])
            for (split_1, split_2) in pairs_to_check
        }
        err_msg: str = "Image(s) are being shared between splits.\n"
        for split_names, overlapping_images in overlap.items():
            if not overlapping_images:
                continue
            err_msg += f"Images in split {split_names[0]} were found in split {split_names[1]}.\n"
            err_msg += "Images found in both sets:\n"
            for im_id in overlapping_images:
                err_msg += f"{im_id}\n"

        raise OverlappingSplitsError(err_msg)

    @model_validator(mode="after")
    def check_splits_span_whole_dataset(self) -> Self:
        """Ensures that every annotated image belongs to a split."""
        all_image_ids: set[ImageId] = {img.image_id for img in self.annotated_images}

        image_ids_in_splits: set[ImageId] = set(
            reduce(set.union, [v for v in self.splits.values()], set())
        )
        image_ids_missing_from_splits: set[ImageId] = (
            all_image_ids - image_ids_in_splits
        )

        if image_ids_missing_from_splits:
            raise DatasetError(
                "Splits provided do not span the whole dataset. "
                + f"Image ids not in a split: {image_ids_missing_from_splits}"
            )

        return self

    @model_validator(mode="after")
    def warn_if_dataset_is_empty(self) -> Self:
        """Warns the user if the dataset is empty."""
        if self.annotated_images == []:
            warnings.warn("Dataset is empty.", EmptyDatasetWarning)

        return self

    @model_validator(mode="after")
    def warn_if_splits_are_empty(self) -> Self:
        """Warns the user if a split is empty."""
        # Avoid redundant warnings.
        if self.annotated_images == []:
            return self

        for split_name, images_in_split in self.splits.items():
            if images_in_split:
                continue
            warnings.warn(f"{split_name} exists but is empty.", EmptySplitWarning)

        return self
