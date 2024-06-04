"""Compile modules for source finding"""

from continunet.image.fits import ImageSquare
from continunet.image.pre_processing import PreProcessor
from continunet.network.unet import Unet


class Finder:
    """Class for source finding in radio continuum images."""

    def __init__(self, image, sources, layers: int = 4):
        # TODO: must be fits object, add check
        self.image = image
        self.layers = layers
        self.sources = sources

        self.find_sources()

    def find_sources(self):
        """Find sources in a continuum image."""
        image_object = ImageSquare(self.image)
        processed_image = PreProcessor(image_object, self.layers).image

        unet = Unet(processed_image, layers=self.layers)

        self.sources = unet.reconstructed

        return self
