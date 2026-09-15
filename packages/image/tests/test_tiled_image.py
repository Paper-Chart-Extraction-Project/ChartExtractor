"""A series of tests for the TiledImage class."""

import numpy as np
from core.geometry import Rectangle

from image import Image
from tile import TiledImage, TilingStrategy


def test_create_tile():
    """Tests the _create_tile static method."""
    im_data: np.ndarray = np.array(
        [[pix * power for pix in [0, 1, 2, 3, 4, 5]] for power in range(6)]
    )
    im: Image = Image.from_cv2(im_data, convert_bgr_to_rgb=False)
    cropped_region: Rectangle = Rectangle.from_left_top_right_bottom(
        left=1, top=1, right=4, bottom=3
    )

    crop: np.ndarray = TiledImage._create_tile(im, cropped_region)
    true_crop: np.ndarray = np.array([[1, 2, 3], [2, 4, 6]])

    assert np.array_equal(crop, true_crop)


def test_tiled_image():
    """Tests the TiledImage class."""
    im_data: np.ndarray = np.array(
        [[pix * power for pix in [0, 1, 2, 3, 4, 5]] for power in range(6)]
    )
    im: Image = Image.from_cv2(im_data, convert_bgr_to_rgb=False)
    tiled_im: TiledImage = TiledImage.from_tile_size_pixel_overlap(
        image=im,
        tile_width=4,
        tile_height=4,
        x_pixel_overlap=2,
        y_pixel_overlap=2,
        tile_strategy=TilingStrategy.EXACT,
    )

    tile_0: np.ndarray = np.array(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]]
    )
    tile_1: np.ndarray = np.array(
        [[0, 0, 0, 0], [2, 3, 4, 5], [4, 6, 8, 10], [6, 9, 12, 15]]
    )
    tile_2: np.ndarray = np.array(
        [[0, 2, 4, 6], [0, 3, 6, 9], [0, 4, 8, 12], [0, 5, 10, 15]]
    )
    tile_3: np.ndarray = np.array(
        [[4, 6, 8, 10], [6, 9, 12, 15], [8, 12, 16, 20], [10, 15, 20, 25]]
    )
    true_tiles: list[list[np.ndarray]] = np.array([[tile_0, tile_1], [tile_2, tile_3]])

    assert np.array_equal(tiled_im.tiles, true_tiles)
