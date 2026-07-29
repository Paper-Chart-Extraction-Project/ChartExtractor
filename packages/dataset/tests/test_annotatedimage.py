"""A collection of series of tests for AnnotatedImage."""

from dataset import AnnotatedImage
from image import ImageMetadata
import pytest
import warnings


@pytest.fixture
def image_metadata() -> ImageMetadata:
    """An ImageMetadata object for testing."""
    return ImageMetadata(path=None)


class TestImageIdWarning:
    """A group of tests that ensures that the UserWarning for ImageId works appropriately."""

    def test_warns_if_no_image_id_supplied(self, image_metadata: ImageMetadata):
        """Checks if the user is warned when an AnnotatedImage is created with no image_id."""
        with pytest.warns(UserWarning, match="image_id"):
            AnnotatedImage(image=image_metadata)

    def test_does_not_warn_if_image_id_supplied(self, image_metadata: ImageMetadata):
        """Checks if the user is *not* warned when an AnnotatedImage is created with an image_id."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            AnnotatedImage(image=image_metadata, image_id="test-test-123")
