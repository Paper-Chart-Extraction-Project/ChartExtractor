"""A module with test fixtures shared between files."""

import pytest
from image import ImageMetadata


@pytest.fixture
def dummy_image_metadata() -> ImageMetadata:
    """An ImageMetadata object for testing."""
    return ImageMetadata(path=None)
