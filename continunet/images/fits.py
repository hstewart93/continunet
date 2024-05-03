"""Models for pre-processing FITS image data."""

from abc import ABC
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


class FitsImage(ABC):
    """Abstract model for an image imported from FITS format."""

    def __init__(self, path, data=None, header=None, wcs=None, beam_size=None, shape=None):
        self.path = path
        self.data = data
        self.header = header
        self.wcs = wcs
        self.beam_size = beam_size
        self.shape = shape

        self.load()

    def load(self):
        """Load fits image from file and populate model args."""
        if not self.path:
            raise ValueError("Path to FITS file not provided.")

        with fits.open(self.path) as fits_object:

            if self.data is None:
                self.data = fits_object[0].data
                # Convert byte ordering to little-endian as FITS is stored as big-endian
                # and is incompatible with torch
                self.data = self.data.astype(np.float32)

            if self.header is None:
                self.header = fits_object[0].header

            if self.wcs is None:
                self.wcs = WCS(self.header)

            if self.beam_size is None:
                self.beam_size = self.get_beam_size()

            if self.shape is None:
                self.shape = self.data.shape

        return self

    def get_beam_size(self):
        """Return the beam size in arcseconds."""
        if "BMAJ" not in self.header:
            raise KeyError("Header does not contain 'BMAJ' (beam size information).")
        if "BMIN" not in self.header:
            raise KeyError("Header does not contain 'BMIN' (beam size information).")

        circular = np.isclose(self.header["BMAJ"] / self.header["BMIN"], 1.0)
        if circular:
            return self.header["BMAJ"] * 3600

        raise KeyError(f"Beam is not circular (BMAJ / BMIN = {circular} =! 1).")


class ImageSquare(FitsImage):
    """Model for a 2D FITS image."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
