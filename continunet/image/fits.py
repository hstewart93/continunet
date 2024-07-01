"""Models for pre-processing FITS image data."""

from abc import ABC
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from continunet.constants import CYAN, RESET


class FitsImage(ABC):
    """Abstract model for an image imported from FITS format."""

    def __init__(self, path):
        self.path = path
        self.data = None
        self.header = None
        self.wcs = None
        self.beam_size = None
        self.shape = None

        self.load()

    def load(self):
        """Load fits image from file and populate model args."""
        print(f"{CYAN}Loading FITS image from {self.path}...{RESET}")
        if not self.path:
            raise ValueError("Path to FITS file not provided.")

        with fits.open(self.path) as fits_object:

            self.data = fits_object[0].data
            # Convert byte ordering to little-endian as FITS is stored as big-endian
            # and is incompatible with torch
            self.data = self.data.astype(np.float32)
            self.header = fits_object[0].header
            self.wcs = WCS(self.header)
            if not self.wcs.has_celestial:
                raise ValueError("WCS object does not contain celestial information.")
            self.beam_size = self.get_beam_size()
            self.shape = self.data.shape
            self.check_header()

        return self

    def check_header(self):
        """Check the header contains required information."""
        required_keys = ["CRPIX1", "CRPIX2"]
        for key in required_keys:
            if key not in self.header:
                raise KeyError(f"Header does not contain '{key}' (image information).")

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
