"""A module with functions for computing the coordinates of tiles when tiling images."""

from enum import StrEnum
from typing import Self

from core.geometry import Rectangle
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


def compute_tile_coordinates(
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    x_pixel_overlap: int,
    y_pixel_overlap: int,
    tiling_strategy: TilingStrategy,
) -> list[Rectangle]:
    """Computes the coordinates for tiles on an image given the tiling parameters.

    This function just builds the tiling context and dispatches it to a function that computes
    the tile coordinates for a specific strategy.

    Args:
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

    Returns:
        The coordinates of the tiles according to the input parameters and the tiling strategy.
    """
    tiling_context: _TilingContext = _TilingContext(
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        tile_height=tile_height,
        x_pixel_overlap=x_pixel_overlap,
        y_pixel_overlap=y_pixel_overlap,
        tiling_strategy=tiling_strategy,
    )

    match tiling_strategy:
        case TilingStrategy.EXACT:
            return compute_exact_tile_coordinates(tiling_context)
        case TilingStrategy.CLAMP:
            return compute_clamp_tile_coordinates(tiling_context)
        case _:
            raise ValueError(
                f"No implementation yet for tiling under the {tiling_strategy} strategy."
            )


def compute_exact_tile_coordinates(tiling_context: _TilingContext) -> list[Rectangle]:
    """Computes the coordinates for tiles on an image under the EXACT tiling strategy.

    Args:
        tiling_context (_TilingContext):
            The paramters for tiling the image.

    Returns:
        A list of rectangles with the coordinates of the tiles.
        Use rect.to_left_top_right_bottom() to get a tuple of coordinates.
    """
    tile_top_sides: range = range(
        0,  # start
        tiling_context.image_height - tiling_context.tile_height + 1,  # stop
        tiling_context.y_pixel_overlap,  # step
    )
    tile_left_sides: range = range(
        0,  # start
        tiling_context.image_width - tiling_context.tile_width + 1,  # stop
        tiling_context.x_pixel_overlap,  # step
    )

    # orders the tiles by row top to bottom, then column left to right.
    return [
        [
            Rectangle.from_top_left_xywh(
                left, top, tiling_context.tile_width, tiling_context.tile_height
            )
            for left in tile_left_sides
        ]
        for top in tile_top_sides
    ]


def _get_clamp_offsets(length: int, tile_len: int, step: int) -> list[int]:
    """Helper to compute 1D origin offsets for clamping tile coordinates to image boundaries.

    In essence, this helper function computes the left or top of tiles under the CLAMP strategy.

    For clarity, the functions _compute_clamp_tile_left_sides and _compute_clamp_tile_top_sides
    both exist which just call this function with the tiling context.

    Args:
        length (int):
            The length of the 1d array to clamp tile.
        tile_len (int):
            The length of a 1d tile.
        step (int):
            The overlap that two 1d tiles should have (except the final tile).

    Returns:
        A list containing the start coordinate of 1d tiles under the CLAMP strategy.
    """
    # The final tile is the image boundary shifted in by the tile size.
    final_tile_position: int = length - tile_len
    # The offsets are computed as normal until the last tile, which is appended.
    return list(range(0, final_tile_position, step)) + [final_tile_position]


def _compute_clamp_tile_left_sides(tiling_context: _TilingContext) -> list[int]:
    """Calls the _get_clamp_offsets function to get the left side of all the tiles."""
    return _get_clamp_offsets(
        length=tiling_context.image_width,
        tile_len=tiling_context.tile_width,
        step=tiling_context.x_pixel_overlap,
    )


def _compute_clamp_tile_top_sides(tiling_context: _TilingContext) -> list[int]:
    """Calls the _get_clamp_offsets function to get the top side of all the tiles."""
    return _get_clamp_offsets(
        length=tiling_context.image_height,
        tile_len=tiling_context.tile_height,
        step=tiling_context.y_pixel_overlap,
    )


def compute_clamp_tile_coordinates(tiling_context: _TilingContext) -> list[Rectangle]:
    """Computes the coordinates for tiles on an image under the EXACT tiling strategy.

    Args:
        tiling_context (_TilingContext):
            The paramters for tiling the image.

    Returns:
        A list of tuples with the left, top, right, and bottom of the tiles.
    """
    tile_top_sides: list[int] = _compute_clamp_tile_top_sides(tiling_context)
    tile_left_sides: list[int] = _compute_clamp_tile_left_sides(tiling_context)

    return [
        [
            Rectangle.from_top_left_xywh(
                left=left,
                top=top,
                width=tiling_context.tile_width,
                height=tiling_context.tile_height,
            )
            for left in tile_left_sides
        ]
        for top in tile_top_sides
    ]
