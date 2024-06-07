"""Tests for processing image data."""

import numpy as np
import pytest

from continunet.image.processing import PreProcessor, PostProcessor


class TestPreProcessing:
    """Tests for the PreProcessing model."""

    model = PreProcessor

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
        # test wcs shape
        assert image.wcs.array_shape == valid_image_shape[1:3]

    def test_reshape_wcs(self, valid_image_object, valid_image_shape):
        """Test the wcs object shape."""
        image = self.model(valid_image_object)
        image.reshape()
        assert image.wcs.array_shape == valid_image_shape[1:3]

    def test_normalise(self, valid_image_object):
        """Test image normalisation."""
        image = self.model(valid_image_object)
        image.normalise()

        assert image.data.min() == 0
        assert image.data.max() == 1

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
        assert image.data.min() == 0
        assert image.data.max() == 1
        assert not np.isnan(image.data).any()
        assert image.wcs.array_shape == valid_image_shape[1:3]


class TestPostProcessing:
    """Tests for the PostProcessing model."""

    model = PostProcessor
