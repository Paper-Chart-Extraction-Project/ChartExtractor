"""A module with functions for computing the coordinates of tiles when tiling images."""

from enum import Enum


class TilingStrategy(Enum):
    """An enum for the different ways to compute tiles."""

    EXACT = 0
    CLAMP = 1
