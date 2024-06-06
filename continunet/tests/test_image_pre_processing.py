"""Tests for pre-processing image data."""

import numpy as np
import pytest

from continunet.image.pre_processing import PreProcessor


class TestPreProcessing:
    """Test suite for the PreProcessing model."""

    model = PreProcessor

    def test_reshape_valid_image(self, valid_image_object, valid_image_shape):
        """Test reshape image method for a valid image shape."""
        image = self.model(valid_image_object)

        assert image.data.shape == valid_image_shape

    def test_reshape_invalid_image(self, invalid_image_object, valid_image_shape):
        """Test reshape image method for an invalid image shape."""
        image = self.model(invalid_image_object)

        assert image.data.shape == valid_image_shape
        # test wcs object shape

    def test_reshape_wcs(self, valid_image_object):
        """Test the wcs object shape."""
        image = self.model(valid_image_object)
        assert image.wcs.array_shape == np.squeeze(image.data).shape

    def test_normalise(self, valid_image_object):
        """Test image normalisation."""
        image = self.model(valid_image_object)

        assert image.data.min() == 0
        assert image.data.max() == 1

    def test_clean_nans(self, nan_image_object):
        """Test cleaning NaNs from the image data."""
        image = self.model(nan_image_object)
        # import ipdb; ipdb.set_trace(context=25)
        assert not np.isnan(image.data).any()

        assert image.data[0, 0, 0] == 0

    def test_clean_nans_all_nans(self, image_object_all_nans):
        """Test cleaning NaNs from the image data."""
        with pytest.raises(ValueError):
            self.model(image_object_all_nans)
