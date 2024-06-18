"""Compile ContinUNet modules into Finder class for source finding."""

import time

from astropy.table import Table

from continunet.constants import GREEN, RESET, TRAINED_MODEL
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
        self.post_processor = None
        self.segmentation_map = None
        self.model_map = None
        self.residuals = None
        self.raw_sources = None

    def find(self, generate_maps=False, use_raw=False):
        """Find sources in a continuum image."""
        start_time = time.time()
        # Load image
        image_object = ImageSquare(self.image)

        # Pre-process image
        pre_processor = PreProcessor(image_object, self.layers)
        data = pre_processor.process()

        # Run U-Net
        unet = Unet(data.shape[1:4], trained_model=TRAINED_MODEL, image=data, layers=self.layers)
        self.reconstructed_image = unet.decode_image()

        # Post-process reconstructed image
        self.post_processor = PostProcessor(unet.reconstructed, pre_processor)
        self.sources = self.post_processor.get_sources()
        self.segmentation_map = self.post_processor.segmentation_map
        self.raw_sources = self.post_processor.raw_sources

        end_time = time.time()
        print(
            f"{GREEN}ContinUnet found {len(self.sources)} sources "
            f"in {(end_time - start_time):.2f} seconds.{RESET}"
        )

        if generate_maps:
            self.model_map = self.post_processor.get_model_map(use_raw)
            self.residuals = self.post_processor.get_residuals(use_raw)

        return self.sources

    def export_sources(self, path: str, export_fits=False):
        """Export source catalogue to a directory."""
        if self.sources is None:
            raise ValueError("No sources to export.")
        if export_fits:
            table = Table.from_pandas(self.sources)
            table.write(path, format="fits", overwrite=True)
            return self

        self.sources.to_csv(path)
        return self
