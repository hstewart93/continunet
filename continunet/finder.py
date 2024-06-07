"""Compile ContinUNet modules into Finder class for source finding."""

from continunet.constants import TRAINED_MODEL
from continunet.image.fits import ImageSquare
from continunet.image.processing import PreProcessor, PostProcessor

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

    def find(self, generate_maps=False, use_raw=False):
        """Find sources in a continuum image."""
        # Load image
        image_object = ImageSquare(self.image)

        # Pre-process image
        pre_processor = PreProcessor(image_object, self.layers)
        data = pre_processor.process()

        # Run U-Net
        unet = Unet(data.shape[1:4], trained_model=TRAINED_MODEL, image=data, layers=self.layers)
        self.reconstructed_image = unet.decode_image()

        # Post-process reconstructed image
        post_processor = PostProcessor(unet.reconstructed, pre_processor)
        self.sources = post_processor.get_sources()
        self.segmentation_map = post_processor.segmentation_map
        self.raw_sources = post_processor.raw_sources

        if generate_maps:
            self.model_map = post_processor.get_model_map(use_raw)
            self.residuals = post_processor.get_residuals(use_raw)

        return self.sources

    def export_sources(self, path: str):
        """Export source catalogue to a directory."""
        if self.sources is None:
            raise ValueError("No sources to export.")
        self.sources.to_csv(path)

        # TODO: export as fits
        return self

    def export_raw_sources(self, path: str):
        """Export raw source catalogue to a directory."""
        if self.raw_sources is None:
            raise ValueError("No raw sources to export.")
        self.raw_sources.to_csv(path)

    def export_reconstructed_image(self, path: str):
        """Export the reconstructed image to a directory."""
        if self.reconstructed_image is None:
            raise ValueError("No reconstructed image to export.")
        # save as numpy array and as image, use custom colour map
        return self

    def export_segmentation_map(self, path: str):
        """Export the segmentation map to a directory."""
        if self.segmentation_map is None:
            raise ValueError("No segmentation map to export.")
        # save as numpy array and as image, use custom colour map
        return self

    def export_model_map(self, path: str):
        """Export the model map to a directory."""
        if self.model_map is None:
            raise ValueError("No model map to export.")
        # save as numpy array and as image, use custom colour map
        return self

    def export_residuals(self, path: str):
        """Export the residuals to a directory."""
        if self.residuals is None:
            raise ValueError("No residuals to export.")
        # save as numpy array and as image, use custom colour map
        return self

    def export(self, path: str):
        """Export all outputs to a directory."""
        self.export_sources(path)
        self.export_reconstructed_image(path)
        self.export_segmentation_map(path)
        self.export_model_map(path)
        self.export_residuals(path)
        return self
