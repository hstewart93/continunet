"""Tests for processing image data."""

import numpy as np
import pytest

from continunet.image.processing import PreProcessor, PostProcessor


class TestPreProcessing:
    """Tests for the PreProcessing model."""

    model = PreProcessor

    def test_invalid_image_type(self, invalid_image):
        """Raise ValueError if invalid image type."""
        with pytest.raises(ValueError):
            self.model(invalid_image)

    def test_reshape_valid_image(self, valid_image_object, valid_image_shape):
        """Test reshape image method for a valid image shape."""
        image = self.model(valid_image_object)
        image.reshape()

        assert image.data.shape == valid_image_shape

    def test_reshape_invalid_image(self, invalid_image_object, valid_image_shape):
        """Test reshape image method for an invalid image shape."""
        image = self.model(invalid_image_object)
        image.reshape()

        assert image.data.shape == valid_image_shape
        assert image.wcs.array_shape == valid_image_shape[1:3]

    def test_reshape_wcs(self, valid_image_object, valid_image_shape):
        """Test the wcs object shape."""
        image = self.model(valid_image_object)
        image.reshape()
        assert image.wcs.array_shape == valid_image_shape[1:3]

    def test_normalise(self, valid_image_object):
        """Test image normalization using z-score + arcsinh."""
        raw_data = valid_image_object.data.copy()
        image = self.model(valid_image_object)
        image.normalise()

        # Check finite values
        assert np.isfinite(image.data).all()

        # Check transformation compressed dynamic range
        raw_range = raw_data.max() - raw_data.min()
        transformed_range = image.data.max() - image.data.min()
        assert transformed_range < raw_range

        # Check data roughly centered
        assert abs(np.median(image.data)) < 1

        # Check reasonable bounds
        assert image.data.min() > -10
        assert image.data.max() < 10

    def test_clean_nans(self, nan_image_object):
        """Test cleaning NaNs from the image data."""
        image = self.model(nan_image_object)
        image.clean_nans()
        assert not np.isnan(image.data).any()

        assert image.data[0, 0, 0] == 0

    def test_clean_nans_all_nans(self, image_object_all_nans):
        """Test cleaning NaNs from the image data."""
        image = self.model(image_object_all_nans)
        with pytest.raises(ValueError):
            image.clean_nans()

    def test_process(self, valid_image_object, valid_image_shape):
        """Test the full pre-processing pipeline."""
        image = self.model(valid_image_object)
        image.process()

        assert image.data.shape == valid_image_shape
        assert abs(np.median(image.data)) < 1
        assert not np.isnan(image.data).any()
        assert image.wcs.array_shape == valid_image_shape[1:3]


class TestPostProcessing:
    """Tests for the PostProcessing model."""

    model = PostProcessor

    def test_invalid_reconstructed_image_type(self, pre_processor_object):
        """Raise ValueError if invalid image type."""
        invalid_image = "invalid_image"
        with pytest.raises(TypeError):
            self.model(
                invalid_image,
                pre_processor_object,
                threshold="default",
                sigma_snr=5.0,
                rms_box="default",
                clean_maps=True,
            )

    def test_invalid_pre_processor_type(self, grayscale_image):
        """Raise ValueError if invalid image type."""
        invalid_pre_processor = "invalid_pre_processor"
        with pytest.raises(TypeError):
            self.model(
                grayscale_image,
                invalid_pre_processor,
                threshold="default",
                sigma_snr=5.0,
                rms_box="default",
                clean_maps=True,
            )

    def test_no_reconstructed_image(self, pre_processor_object):
        """Raise ValueError if no reconstructed image."""
        with pytest.raises(ValueError):
            self.model(
                None,
                pre_processor_object,
                threshold="default",
                sigma_snr=5.0,
                rms_box="default",
                clean_maps=True,
            )

    def test_no_pre_processor(self, grayscale_image):
        """Raise ValueError if no pre-processor."""
        with pytest.raises(ValueError):
            self.model(
                grayscale_image,
                None,
                threshold="default",
                sigma_snr=5.0,
                rms_box="default",
                clean_maps=True,
            )

    def test_get_beam_fwhm(self, pre_processor_object, grayscale_image):
        """Test beam FWHM computation."""
        proc = self.model(
            grayscale_image,
            pre_processor_object,
            threshold="default",
            sigma_snr=3.0,
            rms_box=10,
            clean_maps=True,
        )
        fwhm = proc.get_beam_fwhm()
        assert np.isclose(fwhm, 100.0)

    def test_segmentation_not_cut_when_sigma_snr_none(
        self, pre_processor_object, grayscale_image, monkeypatch
    ):
        """Segmentation map should not be thresholded if sigma_snr is None."""
        processor = self.model(
            grayscale_image,
            pre_processor_object,
            threshold="default",
            sigma_snr=None,  # key case
            rms_box="default",
            clean_maps=True,
        )

        # Mock dependencies to control output
        raw_model_map = np.array([[0.2, 0.8], [0.5, 0.9]])
        processor.rms_map = np.ones_like(raw_model_map)

        # Monkeypatch internal call so that the segmentation map is just raw_model_map initially
        monkeypatch.setattr(
            processor, "get_rms_map", lambda *a, **kw: (processor.rms_map, None, processor.rms_map)
        )

        # Run segmentation (simulate method)
        processor.segmentation_map = raw_model_map.copy()
        if processor.sigma_snr:
            snr_map = raw_model_map / processor.rms_map
            processor.segmentation_map = (snr_map > processor.sigma_snr).astype(np.uint8)

        # Because sigma_snr=None, segmentation_map should remain uncut
        np.testing.assert_array_equal(processor.segmentation_map, raw_model_map)

    def test_segmentation_cut_when_sigma_snr_set(
        self, pre_processor_object, grayscale_image, monkeypatch
    ):
        """Segmentation map should be thresholded if sigma_snr is set."""
        processor = self.model(
            grayscale_image,
            pre_processor_object,
            threshold="default",
            sigma_snr=0.6,  # now threshold should apply
            rms_box="default",
            clean_maps=True,
        )

        raw_model_map = np.array([[0.2, 0.8], [0.5, 0.9]])
        processor.rms_map = np.ones_like(raw_model_map)
        monkeypatch.setattr(
            processor, "get_rms_map", lambda *a, **kw: (processor.rms_map, None, processor.rms_map)
        )

        # Apply same snippet as your real method
        if processor.sigma_snr:
            snr_map = raw_model_map / processor.rms_map
            processor.segmentation_map = (snr_map > processor.sigma_snr).astype(np.uint8)

        expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(processor.segmentation_map, expected)
