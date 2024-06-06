"""Pre-processing module for images."""

import numpy as np

from astropy.nddata import Cutout2D


class PreProcessor:
    """Pre-process image data for inference."""

    def __init__(self, image: object, layers: int = 4):
        self.image = image
        self.layers = layers
        self.data = self.image.data
        self.wcs = self.image.wcs

    def clean_nans(self):
        """Check for NaNs in the image data."""
        if np.isnan(self.data).all():
            raise ValueError("Image data contains only NaNs.")
        if np.isnan(self.data).any():
            self.data = np.nan_to_num(self.data, False)
        return self

    def reshape(self):
        """Reshape the image data for the network. Shape must be divisible by 2 ** n layers."""

        self.data = np.squeeze(self.data)
        self.wcs = self.wcs.celestial
        if not isinstance(self.data.shape[0] / 2 ** self.layers, int) or not isinstance(
            self.data.shape[1] / 2 ** self.layers, int
        ):
            minimum_size = self.data.shape[0] // (2 ** self.layers) * (2 ** self.layers)
            trimmed_image = Cutout2D(
                self.data,
                (self.image.header["CRPIX1"], self.image.header["CRPIX2"]),
                (minimum_size, minimum_size),
                wcs=self.wcs,
            )
            self.data = trimmed_image.data
            self.wcs = trimmed_image.wcs

        self.data = self.data.reshape(1, *self.data.shape, 1)
        return self

    def normalise(self):
        """Normalise the image data."""
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self

    def process(self):
        """Process the image data."""
        self.clean_nans()
        self.reshape()
        self.normalise()
        return self
