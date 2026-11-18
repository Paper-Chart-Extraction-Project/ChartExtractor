"""A module for shared functions that assist in testing."""

from image import ImageMetadata

from dataset import AnnotatedImage, ImageId


def generate_annotated_image(
    image_id: ImageId, dummy_image_metadata: ImageMetadata
) -> AnnotatedImage:
    """A function that generates 'dummy' annotated images.

    Args:
        image_id (ImageId):
            The id to assign the image.

    Returns:
        An AnnotatedImage with no real data and the supplied ImageId.
    """
    return AnnotatedImage(image=dummy_image_metadata, image_id=image_id)
