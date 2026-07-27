"""A short utilities file for tests."""

from pathlib import Path

import pytest

TEST_DATA_DIR: Path = Path(__file__).resolve().parent / "data"


@pytest.fixture
def three_by_three_rgb_test_path() -> Path:
    """The filepath to the 3x3 rgb test image."""
    return TEST_DATA_DIR / "rgb_3x3_test.png"


@pytest.fixture
def four_by_three_black_test_path() -> Path:
    """The filepath to the 3x4 black test image."""
    return TEST_DATA_DIR / "black_4x3_test.jpg"


@pytest.fixture
def four_by_three_exif_rotated_black_test_path() -> Path:
    """The filepath to the 3x4 black test image that has a 90 degree rotation in the exif data."""
    return TEST_DATA_DIR / "black_4x3_test_90_deg_rot_exif.jpg"


@pytest.fixture
def three_by_three_greyscale_test_path() -> Path:
    """The filepath to a single channel greyscale version of the rgb 3x3 test image."""
    return TEST_DATA_DIR / "greyscale_3x3_test.png"
