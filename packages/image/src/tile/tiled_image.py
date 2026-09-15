"""A module containing the TiledImage class."""

from dataclasses import dataclass

import numpy as np
from core.geometry import Rectangle

from image import Image
from tile import TilingStrategy, compute_tile_coordinates


@dataclass(slots=True)
class TiledImage:
    """An image with a list of tile views into the underlying image data.

    Attributes:
        image (Image):
            The image which is tiled.
        tiles (list[np.ndarray.view]):
            The 'tiles' which are numpy array views into cropped sections of the image's data.
    """

    image: Image
    tiles: list[list[np.ndarray]]

    @staticmethod
    def _create_tile(image: Image, tile_coordinates: Rectangle) -> np.ndarray.view:
        """Creates an array view that is a cropped segment of an image (aka: a tile).

        Args:
            image (Image):
                The image to create a tile on.
            tile_coordinates (Rectangle):
                The pixel coordinates to crop. Goes from the pixel on the rectangle's top left
                up to, but not including, the rectangle's bottom right.

        Returns:
            An array view into a segment of the image's data.
        """
        return image._data[
            round(tile_coordinates.top) : round(tile_coordinates.bottom),
            round(tile_coordinates.left) : round(tile_coordinates.right),
        ]

    @classmethod
    def from_tile_size_pixel_overlap(
        cls,
        image: Image,
        tile_width: int,
        tile_height: int,
        x_pixel_overlap: int,
        y_pixel_overlap: int,
        tile_strategy: TilingStrategy = TilingStrategy.CLAMP,
    ) -> TiledImage:
        """Creates a tiled image from an image and set of tiling parameters.

        Args:
            image (Image):
                The image to create a TiledImage from.
            tile_width (int):
                The width that each tile should be.
            tile_height (int):
                The height that each tile should be.
            x_pixel_overlap (int):
                The number of pixels that tiles should overlap left-to-right.
            y_pixel_overlap (int):
                The number of pixels that tiles should overlap top-to-bottom.
            tiling_strategy (TilingStrategy):
                How the tiling should be computed.
                Defaults to 'CLAMP', which tiles each row and column evenly, with the exception
                of the final row or column, which maintains the tile's size, but overlaps with
                the previous row or column more than is specified in order to fully tile the image.
        """
        tile_coordinates: list[Rectangle] = compute_tile_coordinates(
            image_width=image.width,
            image_height=image.height,
            tile_width=tile_width,
            tile_height=tile_height,
            x_pixel_overlap=x_pixel_overlap,
            y_pixel_overlap=y_pixel_overlap,
            tiling_strategy=tile_strategy,
        )
        tiles: list[list[np.ndarray]] = [
            [cls._create_tile(image, tile_coords) for tile_coords in row]
            for row in tile_coordinates
        ]
        return TiledImage(image=image, tiles=tiles)
