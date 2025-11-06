"""Processing module for pre-processing input images and post-processing output
images from the network."""

import math
import numpy as np
import pandas as pd

from astropy.stats import sigma_clip
from astropy.modeling.functional_models import Gaussian2D
from astropy.nddata import Cutout2D
from scipy import ndimage, interpolate
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects

from continunet.image.fits import ImageSquare
from continunet.constants import BLUE, CYAN, MAGENTA, RESET


class PreProcessor:
    """Pre-process image data for inference."""

    def __init__(self, image: object, layers: int = 4):
        if not isinstance(image, ImageSquare):
            raise ValueError("Image must be an ImageSquare object.")
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
        height, width = self.data.shape[:2]

        if (height % (2**self.layers)) != 0 or (width % (2**self.layers)) != 0:
            # Compute nearest smaller size that is divisible by 2**layers
            new_height = (height // (2**self.layers)) * (2**self.layers)
            new_width = (width // (2**self.layers)) * (2**self.layers)
            new_shape = (new_height, new_width)

            print(
                f"{MAGENTA}Image dimensions cannot be processed by the network, "
                f"rehsaping image from {self.data.shape} to {new_shape}.{RESET}"
            )
            self.cutout_object = Cutout2D(
                self.data,
                (self.image.header["CRPIX1"], self.image.header["CRPIX2"]),
                new_shape,
                wcs=self.wcs,
            )
            self.data = self.cutout_object.data
            self.wcs = self.cutout_object.wcs

        else:
            # Image shape is already valid, create a dummy Cutout2D for propogating through
            self.cutout_object = Cutout2D(
                self.data,
                (self.image.header["CRPIX1"], self.image.header["CRPIX2"]),
                self.data.shape,
                wcs=self.wcs,
            )

        self.data = self.data.reshape(1, *self.data.shape, 1)
        return self.data

    def normalise(self):
        """Normalise the image data."""
        mean = np.mean(self.data)
        std = np.std(self.data)
        z_scaled_data = (self.data - mean) / std
        self.data = np.arcsinh(z_scaled_data)
        return self.data

    def process(self):
        """Process the image data."""
        print(f"{CYAN}Pre-processing image...{RESET}")
        self.reshape()
        self.clean_nans()
        self.normalise()
        return self.data


class PostProcessor:
    """Post-processes the output of the neural network, generating segmentation
    maps and source catalogues."""

    def __init__(
        self,
        reconstructed_image: np.ndarray,
        pre_processed_image: object,
        threshold,
        sigma_snr,
        rms_box,
        clean_maps,
    ):
        """Initialise the PostProcessor class.

        Parameters
        ----------
        reconstructed_image : np.ndarray
            The reconstructed image from the neural network.
            Must be a numpy array.
        pre_processed_image : object
            The pre-processed image object. Must be a PreProcessor object.
        threshold : str
            The thresholding method to use for the segmentation map.
            Default is "default" which uses the scikit-image triangle threshold.
            To use a custom threshold, provide a float value.
        """
        if reconstructed_image is None:
            raise ValueError("Reconstructed image must be provided.")
        if not isinstance(reconstructed_image, np.ndarray):
            raise TypeError("Reconstructed image must be a numpy array.")
        self.reconstructed_image = reconstructed_image

        if pre_processed_image is None:
            raise ValueError("Pre-processed image must be provided.")
        if not isinstance(pre_processed_image, PreProcessor):
            raise TypeError("Pre-processed image must be a PreProcessor object.")
        self.pre_processed_image = pre_processed_image
        self.threshold = threshold
        self.sigma_snr = sigma_snr
        self.rms_box = rms_box
        self.segmentation_map = None

        self.labelled_map = None
        self.model_map = None
        self.residuals = None
        self.raw_sources = None
        self.sources = None
        self.header = self.pre_processed_image.image.header
        self.cutout_object = self.pre_processed_image.cutout_object
        self.gaussian_beam = None
        self.rms_map = None
        self.clean_maps = True
        self.nan_mask = None

    def get_beam_fwhm(self):
        """Get FWHM of the beam in pixels from the fits header."""
        pixel_angular_size = self.header["CDELT2"]
        beam_fwhm = self.header["BMAJ"]
        return beam_fwhm / pixel_angular_size

    def get_nan_mask(self):
        """Get mask for nan values (MIGHTEE specific)"""
        nan_image = self.cutout_object.data
        nan_mask = np.zeros((nan_image.shape[0], nan_image.shape[1]))
        self.nan_mask = np.where(nan_image != 0, 1, nan_mask)

        return self.nan_mask

    def estimate_noise_map(
        self, image, box_size=100, step_size=30, sigma=3.0, max_iters=5, mask=None
    ):
        """
        Estimate a spatially varying noise (RMS) map similar to PyBDSF.

        Parameters
        ----------
        image : 2D np.ndarray
            The input residual image (float).
        box_size : int
            Size of the local box (in pixels) for RMS estimation.
        step_size : int
            Step between box centers (in pixels). Controls sampling density.
        sigma : float
            Sigma-clipping threshold (number of sigma to clip).
        max_iters : int
            Maximum number of sigma-clipping iterations.
        mask : 2D np.ndarray, optional
            Boolean mask where True = ignore pixel (e.g., known sources).

        Returns
        -------
        rms_map : 2D np.ndarray
            Interpolated local RMS (noise) map.
        mean_map : 2D np.ndarray
            Interpolated local mean (background) map.
        """

        ny, nx = image.shape
        if mask is None:
            mask = np.zeros_like(image, dtype=bool)

        # Coordinates of box centers
        y_centers = np.arange(box_size // 2, ny - box_size // 2 + 1, step_size)
        x_centers = np.arange(box_size // 2, nx - box_size // 2 + 1, step_size)

        rms_grid = np.zeros((len(y_centers), len(x_centers)))
        mean_grid = np.zeros_like(rms_grid)

        # Loop over boxes
        for iy, y0 in enumerate(y_centers):
            for ix, x0 in enumerate(x_centers):
                sub = image[
                    y0 - box_size // 2 : y0 + box_size // 2, x0 - box_size // 2 : x0 + box_size // 2
                ]
                submask = mask[
                    y0 - box_size // 2 : y0 + box_size // 2, x0 - box_size // 2 : x0 + box_size // 2
                ]

                # Exclude masked/NaN values
                vals = sub[~submask & np.isfinite(sub)]
                if len(vals) < 10:
                    rms_grid[iy, ix] = np.nan
                    mean_grid[iy, ix] = np.nan
                    continue

                # Sigma clipping
                clipped = sigma_clip(vals, sigma=sigma, maxiters=max_iters)
                mean_grid[iy, ix] = np.nanmean(clipped)
                rms_grid[iy, ix] = np.nanstd(clipped)

        # Coordinates for interpolation
        # xx, yy = np.meshgrid(x_centers, y_centers)
        # valid = np.isfinite(rms_grid)

        # Interpolate onto full image grid
        x_full = np.arange(nx)
        y_full = np.arange(ny)

        interp_func_rms = interpolate.RegularGridInterpolator(
            (y_centers, x_centers), np.nan_to_num(rms_grid), bounds_error=False, fill_value=None
        )  # None = extrapolate instead of NaN

        interp_func_mean = interpolate.RegularGridInterpolator(
            (y_centers, x_centers), np.nan_to_num(mean_grid), bounds_error=False, fill_value=None
        )

        # interp_func_rms = interpolate.RegularGridInterpolator(
        #     (y_centers, x_centers),
        # np.nan_to_num(rms_grid), bounds_error=False, fill_value=np.nan)
        # interp_func_mean = interpolate.RegularGridInterpolator(
        #     (y_centers, x_centers),
        # np.nan_to_num(mean_grid), bounds_error=False, fill_value=np.nan)

        yy_full, xx_full = np.meshgrid(y_full, x_full, indexing="ij")
        points = np.stack((yy_full.ravel(), xx_full.ravel()), axis=-1)

        rms_map = interp_func_rms(points).reshape(image.shape)
        mean_map = interp_func_mean(points).reshape(image.shape)

        # Optional: smooth final maps slightly to reduce interpolation noise
        rms_map = ndimage.gaussian_filter(rms_map, sigma=box_size / 5)
        mean_map = ndimage.gaussian_filter(mean_map, sigma=box_size / 5)

        return rms_map, mean_map

    def get_rms_map(self, rms_box="default"):
        """Make RMS map using residual and model maps."""
        # Get raw model map and residuals
        raw_model_map = self.cutout_object.data * self.segmentation_map
        raw_residuals = self.cutout_object.data - raw_model_map

        print(f"{CYAN}Creating RMS map...{RESET}")
        if rms_box == "default":
            rms_box = math.floor(self.get_beam_fwhm()) * 10
        rms_map, _ = self.estimate_noise_map(raw_residuals, box_size=rms_box)
        self.get_nan_mask()
        self.rms_map = np.where(
            self.nan_mask == 0,
            np.nan,
            rms_map,
        )
        return raw_model_map, raw_residuals, self.rms_map

    def get_segmentation_map(self):
        """Calculate the segmentation map from the reconstructed image.
        Only binary segmentation maps are currently supported."""
        print(f"{CYAN}Generating segmentation map...{RESET}")
        if (
            self.threshold != "default"
            and self.threshold != "otsu"
            and not isinstance(self.threshold, float)
        ):
            raise ValueError("Threshold must be 'default', 'otsu', or a float value.")
        if self.threshold == "default":
            print(
                f"{BLUE}Using default thresholding method (scikit-image triangle threshold).{RESET}"
            )
            self.threshold = threshold_triangle(self.reconstructed_image)

        if self.threshold == "otsu":
            print(f"{BLUE}Using Otsu thresholding method.{RESET}")
            self.threshold = threshold_otsu(self.reconstructed_image)
        if isinstance(self.threshold, float):
            print(f"{BLUE}Using custom threshold value: {self.threshold}.{RESET}")
        binary = self.reconstructed_image > self.threshold
        self.segmentation_map = binary.astype(int)[0, :, :, 0]

        if self.clean_maps:
            print(f"{CYAN}Removing objects smaller than beam FWHM...{RESET}")
            # remove objects smaller than the beam FWHM
            min_pixels = self.get_beam_fwhm()
            self.segmentation_map = remove_small_objects(
                self.segmentation_map.astype(bool), min_size=min_pixels
            )

            # get rms map
            raw_model_map, _, _ = self.get_rms_map(self.rms_box)

            if self.sigma_snr:
                snr_map = raw_model_map / self.rms_map
                self.segmentation_map = (snr_map > self.sigma_snr).astype(np.uint8)

        return self.segmentation_map

    def get_labelled_map(self):
        """Label the binary segmentation map."""
        self.get_segmentation_map()
        print(f"{CYAN}Labelling sources...{RESET}")
        self.labelled_map = label(self.segmentation_map, connectivity=2)
        return self.labelled_map

    def get_raw_sources(self):
        """Get the raw sources from the labelled map."""
        self.get_labelled_map()
        print(f"{CYAN}Calculating source properties...{RESET}")
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

    @staticmethod
    def convert_orientation_to_position_angle(orientation):
        """Convert the orientation of the source to position angle in degrees."""
        return np.degrees(orientation + (np.pi / 2))

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
        beam_five_sigma_area = np.pi * beam_five_sigma**2
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

        print(f"{CYAN}Correcting source catalogue...{RESET}")
        catalogue = self.raw_sources.copy()
        catalogue = catalogue[
            (catalogue["axis_major_length"] >= 1) & (catalogue["axis_minor_length"] >= 1)
        ]
        catalogue["image_intensity"] = catalogue["image_intensity"].apply(self.sum_array)

        ra, dec = self.cutout_object.wcs.all_pix2world(
            catalogue["centroid-1"], catalogue["centroid-0"], 0
        )
        catalogue["right_ascension"] = ra
        catalogue["declination"] = dec

        area_correction_factor = self.calculate_area_correction_factor()
        catalogue = catalogue[catalogue.image_intensity > 0]
        catalogue["image_intensity"] = catalogue["image_intensity"] * area_correction_factor

        catalogue["area"] = self.calculate_ellipse_area(
            catalogue.axis_major_length / 2, catalogue.axis_minor_length / 2
        )

        catalogue["position_angle"] = self.convert_orientation_to_position_angle(
            catalogue.orientation
        )

        catalogue = self.correct_flux_densities(catalogue, self.segmentation_map)

        # rename and drop columns
        catalogue = catalogue.rename(
            columns={
                "axis_major_length": "major_axis",
                "axis_minor_length": "minor_axis",
                "image_intensity": "flux_density_uncorrected",
                "intensity_sum_corrected": "flux_density",
                "centroid-0": "y_location_cutout",
                "centroid-1": "x_location_cutout",
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
        if self.model_map is None:
            self.get_model_map(use_raw)
        self.residuals = self.cutout_object.data - self.model_map
        return self.residuals
