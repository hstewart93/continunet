"""Processing module for pre-processing input images and post-processing output
images from the network."""

import math
import numpy as np
import pandas as pd

from astropy.modeling.functional_models import Gaussian2D
from astropy.nddata import Cutout2D
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops_table


class PreProcessor:
    """Pre-process image data for inference."""

    def __init__(self, image: object, layers: int = 4):
        self.image = image
        self.layers = layers
        self.data = self.image.data
        self.wcs = self.image.wcs
        self.cutout_object = None

    def clean_nans(self):
        """Check for NaNs in the image data."""
        if np.isnan(self.data).all():
            raise ValueError("Image data contains only NaNs.")
        if np.isnan(self.data).any():
            self.data = np.nan_to_num(self.data, False)
        return self.data

    def reshape(self):
        """Reshape the image data for the network. Shape must be divisible by 2 ** n layers."""

        self.data = np.squeeze(self.data)
        self.wcs = self.wcs.celestial
        if not isinstance(self.data.shape[0] / 2 ** self.layers, int) or not isinstance(
            self.data.shape[1] / 2 ** self.layers, int
        ):
            minimum_size = self.data.shape[0] // (2 ** self.layers) * (2 ** self.layers)
            print(
                "Image dimensions cannot be processed by the network, "
                f"rehsaping image from {self.data.shape} to {(minimum_size, minimum_size)}."
            )
            self.cutout_object = Cutout2D(
                self.data,
                (self.image.header["CRPIX1"], self.image.header["CRPIX2"]),
                (minimum_size, minimum_size),
                wcs=self.wcs,
            )
            self.data = self.cutout_object.data
            self.wcs = self.cutout_object.wcs

        self.data = self.data.reshape(1, *self.data.shape, 1)
        return self.data

    def normalise(self):
        """Normalise the image data."""
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self.data

    def process(self):
        """Process the image data."""
        self.reshape()
        self.clean_nans()
        self.normalise()
        return self.data


class PostProcessor:
    """Post-processes the output of the neural network, generating segmentation
    maps and source catalogues."""

    def __init__(
        self, reconstructed_image: np.ndarray, pre_processed_image: object, threshold="default"
    ):
        if not isinstance(reconstructed_image, np.ndarray):
            raise ValueError("Reconstructed image must be a numpy array.")
        self.reconstructed_image = reconstructed_image
        if not isinstance(pre_processed_image, PreProcessor):
            raise ValueError("Pre-processed image must be a PreProcessor object.")
        self.pre_processed_image = pre_processed_image
        self.threshold = threshold
        self.segmentation_map = None
        self.labelled_map = None
        self.model_map = None
        self.residuals = None
        self.raw_sources = None
        self.sources = None
        self.header = self.pre_processed_image.image.header
        self.cutout_object = self.pre_processed_image.cutout_object
        self.gaussian_beam = None

    def get_segmentation_map(self):
        """Calculate the segmentation map from the reconstructed image.
        Only binary segmentation maps are currently supported."""
        if self.threshold == "default":
            print("Using default thresholding method (scikit-image triangle threshold).")
            threshold = threshold_triangle(self.reconstructed_image)
        else:
            threshold = self.threshold
        binary = self.reconstructed_image > threshold
        self.segmentation_map = binary.astype(int)[0, :, :, 0]
        return self.segmentation_map

    def get_labelled_map(self):
        """Label the binary segmentation map."""
        self.get_segmentation_map()
        self.labelled_map = label(self.segmentation_map, connectivity=2)
        return self.labelled_map

    def get_raw_sources(self):
        """Get the raw sources from the labelled map."""
        self.get_labelled_map()
        properties = [
            "centroid",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
            "coords",
            "image_intensity",
            "label",
            "perimeter",
        ]
        properties_table = regionprops_table(
            self.labelled_map, self.cutout_object.data, properties=properties
        )
        self.raw_sources = pd.DataFrame(properties_table)
        return self.raw_sources

    def calculate_area_correction_factor(self):
        """Function to calculate the area correction factor for a given image."""
        # in arcseconds
        increment = self.header["CDELT2"] * 3600
        beam_bmaj = self.header["BMAJ"] * 3600
        beam_bmin = self.header["BMIN"] * 3600

        return 8 * np.log(2) * (increment * increment) / (beam_bmaj * beam_bmin * 2 * np.pi)

    @staticmethod
    def sum_array(array):
        """Sum the array."""
        return np.sum(array)

    @staticmethod
    def calculate_ellipse_area(bmaj, bmin):
        """Function to calculate the area of an ellipse."""
        return np.pi * bmaj * bmin

    def generate_2d_gaussian_beam(self):
        """Generate a 2D Gaussian beam for a given fits image."""
        sigma_pixels = self.get_beam_size()
        # model beam as 2d gaussian
        image_size = math.ceil(5 * sigma_pixels * 2)

        # Create a 2D Gaussian model
        gaussian_model = Gaussian2D(1.0, image_size / 2, image_size / 2, sigma_pixels, sigma_pixels)

        # Create a grid of coordinates and calculate the Gaussian values at each coordinate
        coordinate_indeces = np.arange(image_size + 1)
        row_coordinates, column_coordinates = np.meshgrid(coordinate_indeces, coordinate_indeces)
        self.gaussian_beam = gaussian_model(row_coordinates, column_coordinates)

        return self.gaussian_beam

    def get_beam_size(self):
        """Get sigma of the beam in pixels from the fits header."""
        pixel_angular_size = self.header["CDELT2"]
        beam_fwhm = self.header["BMAJ"]
        fwhm_pixels = beam_fwhm / pixel_angular_size

        # convert fwhm to sigma
        return fwhm_pixels / np.sqrt(8 * np.log(2))

    def get_source_mask(self, predicted_map, source, beam_shape):
        """Get the mask of the source in the cutout."""
        cutout = Cutout2D(
            predicted_map, (source["centroid-1"], source["centroid-0"]), beam_shape[0]
        )
        mask = np.where(cutout.data != 0, 1, cutout.data)

        # Handle sources at the edge of the cutout by padding the arrays evenly on each side
        return self.pad_to_target_shape(mask, beam_shape)

    @staticmethod
    def pad_to_target_shape(array, target_shape):
        """Pad the array to match the target shape."""
        if array.shape == target_shape:
            return array

        row_padding = (target_shape[0] - array.shape[0]) // 2
        col_padding = (target_shape[1] - array.shape[1]) // 2
        padded_array = np.pad(
            array, ((row_padding, row_padding), (col_padding, col_padding)), mode="constant"
        )

        # dirty fix for masks of shape (10, 13) and (13, 10)
        if padded_array.shape[0] != target_shape[0]:
            padded_array = np.pad(padded_array, ((0, 1), (0, 0)), mode="constant")
        if padded_array.shape[1] != target_shape[1]:
            padded_array = np.pad(padded_array, ((0, 0), (0, 1)), mode="constant")
        return padded_array

    def correct_flux_densities(self, properties, predicted_map):
        """Correct the flux densities of the sources in the cutout
        for undersampling the synthesized beam."""
        beam = self.generate_2d_gaussian_beam()
        beam_five_sigma = 5 * self.get_beam_size()
        beam_five_sigma_area = np.pi * beam_five_sigma ** 2
        for source_index, source in properties.iterrows():
            if source.area > beam_five_sigma_area:
                correction_factor = 1
                properties.at[source_index, "correction_factor"] = correction_factor

            mask = self.get_source_mask(predicted_map, source, beam.shape)

            masked_beam = np.sum(beam * mask)
            correction_factor = np.sum(beam) / masked_beam
            properties.at[source_index, "correction_factor"] = correction_factor
            properties.at[source_index, "intensity_sum_corrected"] = (
                source.image_intensity * correction_factor
            )
        return properties

    def get_sources(self):
        """Clean the raw sources to produce a catalogue of sources."""
        self.get_raw_sources()

        catalogue = self.raw_sources.copy()
        catalogue["image_intensity"] = catalogue["image_intensity"].apply(self.sum_array)

        catalogue["x_location"] = catalogue["centroid-1"] + self.cutout_object.xmin_original
        catalogue["y_location"] = catalogue["centroid-0"] + self.cutout_object.ymin_original

        wcs_object = self.pre_processed_image.wcs
        ra, dec = wcs_object.all_pix2world(catalogue.x_location, catalogue.y_location, 0)
        catalogue["RA"] = ra
        catalogue["Dec"] = dec

        area_correction_factor = self.calculate_area_correction_factor()
        catalogue = catalogue[catalogue.image_intensity > 0]
        catalogue["image_intensity"] = catalogue["image_intensity"] * area_correction_factor

        catalogue["area"] = self.calculate_ellipse_area(
            catalogue.axis_major_length / 2, catalogue.axis_minor_length / 2
        )

        catalogue = catalogue[
            (catalogue["axis_major_length"] >= 1) & (catalogue["axis_minor_length"] >= 1)
        ]

        catalogue = self.correct_flux_densities(catalogue, self.segmentation_map)

        # TODO: convert orientation to degrees as for SDC1

        # rename and drop columns
        catalogue = catalogue.rename(
            columns={
                "axis_major_length": "bmaj",
                "axis_minor_length": "bmin",
                "orientation": "position_angle",
                "image_intensity": "flux_uncorrected",
                "intensity_sum_corrected": "flux",
                "centroid-0": "y_location_original",
                "centroid-1": "x_location_original",
            },
        )
        catalogue = catalogue.drop(columns=["coords", "perimeter"])
        self.sources = catalogue
        return self.sources

    def get_model_map(self, use_raw=False):
        """Calculate the model map from the cleaned segmentation map and the input
        image. If use_raw is True, the raw sources are used to create the model map."""
        if use_raw:
            if self.raw_sources is None:
                self.get_raw_sources()
            return self.cutout_object.data * self.segmentation_map

        if self.sources is None:
            self.get_sources()
        unique_labels = self.sources.label.unique()
        mask = np.isin(self.labelled_map, unique_labels)
        self.segmentation_map *= mask
        self.model_map = self.cutout_object.data * self.segmentation_map
        return self.model_map

    def get_residuals(self, use_raw=False):
        """Calculate the residuals from the input image and the model map."""
        self.get_model_map(use_raw)
        self.residuals = self.cutout_object.data - self.model_map
        return self.residuals
