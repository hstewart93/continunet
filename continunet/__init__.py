"""ContinuNet package."""

from continunet.finder import Finder


def extract_sources(image, layers=4):
    """Extract sources from a continuum image."""
    return Finder(image, layers).sources
