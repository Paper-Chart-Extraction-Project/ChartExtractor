"""A series of tests for the compute_tile_coordinates module."""

import pytest
from core.geometry import Rectangle
from tile.compute_tile_coordinates import (
    TilingStrategy,
    _compute_clamp_tile_coordinates,
    _compute_exact_tile_coordinates,
    _get_clamp_offsets,
    _TilingContext,
)


def generate_tiling_context(
    image_width: int = 100,
    image_height: int = 150,
    tile_width: int = 40,
    tile_height: int = 40,
    x_pixel_overlap: int = 20,
    y_pixel_overlap: int = 10,
    tiling_strategy: TilingStrategy = TilingStrategy.EXACT,
) -> _TilingContext:
    """Generates a tiling context with default parameters."""
    return _TilingContext(
        image_width=image_width,
        image_height=image_height,
        tile_width=tile_width,
        tile_height=tile_height,
        x_pixel_overlap=x_pixel_overlap,
        y_pixel_overlap=y_pixel_overlap,
        tiling_strategy=tiling_strategy,
    )


class TestTilingContext:
    """A group of tests for the _TilingContext class."""

    def test_validate_tiling_parameters_with_valid_inputs(self):
        """Tests the validate_tiling_parameters method with valid inputs."""
        generate_tiling_context()

    def test_validate_tiling_parameters_tile_width_gt_im_width(self):
        """Tests the validate_tiling_parameters where tile width > image width."""
        tile_width: int = 100 + 1

        with pytest.raises(ValueError, match="greater than image width"):
            generate_tiling_context(tile_width=tile_width)

        with pytest.raises(ValueError, match="greater than image width"):
            generate_tiling_context(
                tile_width=tile_width, tiling_strategy=TilingStrategy.CLAMP
            )

    def test_validate_tiling_parameters_tile_height_gt_im_height(self):
        """Tests the validate_tiling_parameters where tile height > image height."""
        tile_height: int = 150 + 1

        with pytest.raises(ValueError, match="greater than image height"):
            generate_tiling_context(tile_height=tile_height)

        with pytest.raises(ValueError, match="greater than image height"):
            generate_tiling_context(
                tile_height=tile_height, tiling_strategy=TilingStrategy.CLAMP
            )

    def test_validate_tiling_parameters_negative_x_px_overlap(self):
        """Tests the validate_tiling_parameters where the x pixel overlap < 0."""
        x_pixel_overlap: int = -1

        with pytest.raises(ValueError, match="negative x pixel overlap"):
            generate_tiling_context(x_pixel_overlap=x_pixel_overlap)

        with pytest.raises(ValueError, match="negative x pixel overlap"):
            generate_tiling_context(
                x_pixel_overlap=x_pixel_overlap, tiling_strategy=TilingStrategy.CLAMP
            )

    def test_validate_tiling_parameters_negative_y_px_overlap(self):
        """Tests the validate_tiling_parameters where the y pixel overlap < 0."""
        y_pixel_overlap: int = -1

        with pytest.raises(ValueError, match="negative y pixel overlap"):
            generate_tiling_context(y_pixel_overlap=y_pixel_overlap)

        with pytest.raises(ValueError, match="negative y pixel overlap"):
            generate_tiling_context(
                y_pixel_overlap=y_pixel_overlap, tiling_strategy=TilingStrategy.CLAMP
            )

    def test_validate_exact_tiling_bad_x_px_overlap(self):
        """Tests the _validate_exact_tiling_parameters with a x px overlap that won't work for exact tiling."""
        x_pixel_overlap: int = 24

        with pytest.raises(ValueError, match="Given tiling parameters will not"):
            generate_tiling_context(x_pixel_overlap=x_pixel_overlap)

    def test_validate_clamp_tiling_nonexact_x_px_overlap(self):
        """Tests the validate_tiling_parameters method with a valid x px overlap

        Tests validate_tiling_parameters with a x pixel overlap that wouldn't work for exact
        tiling, but which will work for clamp tiling. Makes sure that the function doesn't raise an
        error for a valid x pixel overlap.
        """
        x_pixel_overlap: int = 24

        generate_tiling_context(
            x_pixel_overlap=x_pixel_overlap, tiling_strategy=TilingStrategy.CLAMP
        )

    def test_validate_exact_tiling_bad_y_px_overlap(self):
        """Tests the _validate_exact_tiling_parameters with a y px overlap that won't work for exact tiling."""
        y_pixel_overlap: int = 12

        with pytest.raises(ValueError, match="Given tiling parameters will not"):
            generate_tiling_context(y_pixel_overlap=y_pixel_overlap)

    def test_validate_clamp_tiling_nonexact_y_px_overlap(self):
        """Tests the validate_tiling_parameters method with a valid y px overlap

        Tests validate_tiling_parameters with a y pixel overlap that wouldn't work for exact
        tiling, but which will work for clamp tiling. Makes sure that the function doesn't raise an
        error for a valid y pixel overlap.
        """
        y_pixel_overlap: int = 12

        generate_tiling_context(
            y_pixel_overlap=y_pixel_overlap, tiling_strategy=TilingStrategy.CLAMP
        )


def test_compute_exact_tile_coordinates():
    """Tests the compute_exact_tile_coordinates function."""
    tiling_context: _TilingContext = generate_tiling_context(
        tile_width=50, tile_height=100, x_pixel_overlap=25, y_pixel_overlap=50
    )
    tile_coords: list[Rectangle] = _compute_exact_tile_coordinates(tiling_context)
    true_tile_coords: list[Rectangle] = [
        [
            Rectangle.from_left_top_right_bottom(
                left=0.0, top=0.0, right=50.0, bottom=100.0
            ),
            Rectangle.from_left_top_right_bottom(
                left=25.0, top=0.0, right=75.0, bottom=100.0
            ),
            Rectangle.from_left_top_right_bottom(
                left=50.0, top=0.0, right=100.0, bottom=100.0
            ),
        ],
        [
            Rectangle.from_left_top_right_bottom(
                left=0.0, top=50.0, right=50.0, bottom=150.0
            ),
            Rectangle.from_left_top_right_bottom(
                left=25.0, top=50.0, right=75.0, bottom=150.0
            ),
            Rectangle.from_left_top_right_bottom(
                left=50.0, top=50.0, right=100.0, bottom=150.0
            ),
        ],
    ]
    assert tile_coords == true_tile_coords


class TestComputeClampTileCoordinates:
    """A group of tests for the compute_clamp_tile_coordinates function."""

    def test_get_clamp_offsets_exact_tiling(self):
        """Tests the _get_clamp_offsets helper function."""
        clamp_offsets: list[int] = _get_clamp_offsets(100, 25, 25)
        true_clamp_offsets: list[int] = [0, 25, 50, 75]
        assert clamp_offsets == true_clamp_offsets

    def test_get_clamp_offsets_inexact_tiling(self):
        """Tests the _get_clamp_offsets helper function."""
        clamp_offsets: list[int] = _get_clamp_offsets(100, 30, 25)
        true_clamp_offsets: list[int] = [0, 25, 50, 70]
        assert clamp_offsets == true_clamp_offsets

    def test_compute_clamp_tile_coordinates(self):
        """Tests the compute_clamp_tile_coordinates function."""
        tiling_context: _TilingContext = generate_tiling_context(
            tile_width=50,
            tile_height=100,
            x_pixel_overlap=30,
            y_pixel_overlap=50,
            tiling_strategy=TilingStrategy.CLAMP,
        )
        tile_coords: list[Rectangle] = _compute_clamp_tile_coordinates(tiling_context)
        true_tile_coords: list[Rectangle] = [
            [
                Rectangle.from_left_top_right_bottom(
                    left=0.0, top=0.0, right=50.0, bottom=100.0
                ),
                Rectangle.from_left_top_right_bottom(
                    left=30.0, top=0.0, right=80.0, bottom=100.0
                ),
                Rectangle.from_left_top_right_bottom(
                    left=50.0, top=0.0, right=100.0, bottom=100.0
                ),
            ],
            [
                Rectangle.from_left_top_right_bottom(
                    left=0.0, top=50.0, right=50.0, bottom=150.0
                ),
                Rectangle.from_left_top_right_bottom(
                    left=30.0, top=50.0, right=80.0, bottom=150.0
                ),
                Rectangle.from_left_top_right_bottom(
                    left=50.0, top=50.0, right=100.0, bottom=150.0
                ),
            ],
        ]
        assert tile_coords == true_tile_coords
