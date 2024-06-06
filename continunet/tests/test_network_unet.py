"""Tests for the UNet model."""

import pytest

from continunet.constants import TRAINED_MODEL
from continunet.network.unet import Unet


class TestUnet:
    """Tests for the UNet model."""

    model = Unet

    def test_build_model(self, input_shape):
        """Test the compile_model method"""
        test_model = self.model(input_shape)

        assert test_model.model.input_shape == (None, *input_shape)
        assert test_model.model.output_shape == (None, *input_shape)
        assert len(test_model.model.layers) == 49

    def test_build_model_invalid_input_shape(self, invalid_image, invalid_image_input_shape):
        """Test the decode_image method with invalid input shape"""
        with pytest.raises(ValueError):
            self.model(invalid_image_input_shape, image=invalid_image, trained_model=TRAINED_MODEL)

    def test_load_weights(self, input_shape):
        """Test the load_weights method"""

        test_model = self.model(input_shape)
        test_model.model.load_weights(TRAINED_MODEL)

        assert test_model.model.input_shape == (None, *input_shape)
        assert test_model.model.output_shape == (None, *input_shape)
        assert len(test_model.model.layers) == 49
        assert test_model.model.get_weights() is not None

    def test_decode_image(self, grayscale_image, input_shape):
        """Test the decode_image method"""

        test_model = self.model(input_shape, image=grayscale_image, trained_model=TRAINED_MODEL)

        decoded_image = test_model.decode_image()
        assert decoded_image.shape == (1, *input_shape)

        assert decoded_image.min() >= 0
        assert decoded_image.max() <= 1

    def test_decode_image_invalid_image_type(self, input_shape):
        """Test the decode_image method with invalid image type"""
        test_model = self.model(input_shape, image="invalid", trained_model=TRAINED_MODEL)
        with pytest.raises(TypeError):
            test_model.decode_image()

    def test_decode_image_no_trained_model(self, grayscale_image, grayscale_image_input_shape):
        """Test the decode_image method with no trained model"""
        test_model = self.model(grayscale_image_input_shape, image=grayscale_image)
        with pytest.raises(ValueError):
            test_model.decode_image()

    def test_decode_image_no_image(self, input_shape):
        """Test the decode_image method with no image"""
        test_model = self.model(input_shape, trained_model=TRAINED_MODEL)
        with pytest.raises(ValueError):
            test_model.decode_image()

    def test_decode_image_colour_image(self, colour_image, colour_image_input_shape):
        """Test the decode_image method with a colour image"""
        test_model = self.model(
            colour_image_input_shape, image=colour_image, trained_model=TRAINED_MODEL,
        )
        with pytest.raises(ValueError):
            test_model.decode_image()
