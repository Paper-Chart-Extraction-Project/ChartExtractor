"""A module containing custom exceptions for dataset errors."""


class DatasetError(Exception):
    """A generic exception regarding dataset issues."""


class OverlappingSplitsError(DatasetError):
    """An error raised when there are images shared between the 'train', 'val', and 'test' sets."""
