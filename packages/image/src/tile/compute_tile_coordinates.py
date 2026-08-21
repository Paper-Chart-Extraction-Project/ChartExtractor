"""A module with functions for computing the coordinates of tiles when tiling images."""

from enum import StrEnum
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, model_validator


class TilingStrategy(StrEnum):
    """An enum for the different ways to compute tiles."""

    EXACT = "EXACT"
    CLAMP = "CLAMP"


class _TilingContext(BaseModel):
    """A private class that validates and stores tiling parameters.

    Attributes:
        image_width (int):
            The width of the image to tile.
        image_height (int):
            The height of the image to tile.
        tile_width (int):
            The width of a single tile.
        tile_height (int):
            The height of a single tile.
        x_pixel_overlap (int):
            The number of pixels that tiles should overlap left-to-right.
        y_pixel_overlap (int):
            The number of pixels that tiles should overlap top-to-bottom.
        tiling_strategy (TilingStrategy):
            How the tiling should be computed.
    """

    model_config = ConfigDict(frozen=True)

    image_width: int
    image_height: int
    tile_width: int
    tile_height: int
    x_pixel_overlap: int
    y_pixel_overlap: int
    tiling_strategy: TilingStrategy

    @model_validator(mode="after")
    def validate_tiling_parameters(self) -> Self:
        """Determines if the tiling parameters are valid for the image given the tiling strategy.

        Performs validation that is common to all tiling strategies, then dispatches to
        strategy-specific validation methods for the rest, if necessary.
        """

        if self.tile_width > self.image_width:
            raise ValueError(
                f"Tile width ({self.tile_width}) cannot be "
                + f"greater than image width ({self.image_width})."
            )
        if self.tile_height > self.image_height:
            raise ValueError(
                f"Tile height ({self.tile_height}) cannot be "
                + f"greater than image height ({self.image_height})."
            )
        if self.x_pixel_overlap < 0:
            raise ValueError(
                f"Cannot tile with a negative x pixel overlap ({self.x_pixel_overlap})."
            )
        if self.y_pixel_overlap < 0:
            raise ValueError(
                f"Cannot tile with a negative y pixel overlap ({self.y_pixel_overlap})."
            )

        match self.tiling_strategy:
            case TilingStrategy.EXACT:
                self._validate_exact_tiling_parameters()
            case TilingStrategy.CLAMP:
                pass
            case _:
                raise ValueError(
                    "Unable to validate tiling parameters for "
                    + f"tiling strategy {self.tiling_strategy}."
                )

        return self

    def _validate_exact_tiling_parameters(self) -> None:
        """Validates tiling parameters under the EXACT tiling strategy."""
        if (self.image_width - self.tile_width) % self.x_pixel_overlap != 0:
            raise ValueError(
                f"Given tiling parameters will not exactly tile the image.\n{self}"
            )
        if (self.image_height - self.tile_height) % self.y_pixel_overlap != 0:
            raise ValueError(
                f"Given tiling parameters will not exactly tile the image.\n{self}"
            )
