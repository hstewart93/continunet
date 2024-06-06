"""Compile ContinUNet modules into Finder class for source finding."""

from continunet.constants import TRAINED_MODEL
from continunet.image.fits import ImageSquare
from continunet.image.pre_processing import PreProcessor
from continunet.network.unet import Unet


class Finder:
    """Class for source finding in radio continuum images."""

    def __init__(self, image: str, layers: int = 4):
        if not image.endswith(".fits"):
            raise ValueError("File must be a .fits file.")
        self.image = image
        if layers != 4:
            raise ValueError("Number of layers must be 4.")
        self.layers = layers
        self.sources = None
        self.reconstructed_image = None
        self.segmentation_map = None
        self.model_map = None
        self.residuals = None
        self.raw_sources = None

    def extract_sources(self):
        """Find sources in a continuum image."""
        image_object = ImageSquare(self.image)
        pre_processed_image = PreProcessor(image_object, self.layers)
        data = pre_processed_image.process()
        unet = Unet(data.shape[1:4], trained_model=TRAINED_MODEL, image=data, layers=self.layers)
        self.reconstructed_image = unet.decode_image()
        # pre processor is not implemented but should take PreProcessor object and
        # self.reconstructed_image
        # PreProcessor has input data and wcs information for creating catalogue
        # post_processed_image = PostProcessor(unet.reconstructed, pre_processed_image)

        # self.sources = post_processed_image.catalogue

        return self

    def get_model_map(self):
        """Calculate the model map from the cleaned segmentation map
        and the input image."""
        pass

    def get_residuals(self):
        """Calculate the residuals from the input image and the model map."""
        pass
