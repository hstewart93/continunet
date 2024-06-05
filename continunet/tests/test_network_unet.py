"""Tests for the UNet model."""

import pytest

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

    def test_load_weights(self, trained_model, input_shape):
        """Test the load_weights method"""

        test_model = self.model(input_shape)
        test_model.model.load_weights(trained_model)

        assert test_model.model.input_shape == (None, *input_shape)
        assert test_model.model.output_shape == (None, *input_shape)
        assert len(test_model.model.layers) == 49
        assert test_model.model.get_weights() is not None

    def test_decode_image(self, grayscale_image, trained_model, input_shape):
        """Test the decode_image method"""

        test_model = self.model(
            input_shape, image=grayscale_image, trained_model=trained_model, decode=True
        )

        decoded_image = test_model.reconstructed
        assert decoded_image.shape == (1, *input_shape)

        assert decoded_image.min() >= 0
        assert decoded_image.max() <= 1

    def test_decode_image_invalid_image_type(self, trained_model, input_shape):
        """Test the decode_image method with invalid image type"""
        with pytest.raises(TypeError):
            self.model(input_shape, image="invalid", trained_model=trained_model, decode=True)

    def test_decode_image_invalid_input_shape(
        self, invalid_image, trained_model, invalid_image_input_shape
    ):
        """Test the decode_image method with invalid input shape"""
        with pytest.raises(ValueError):
            self.model(
                invalid_image_input_shape,
                image=invalid_image,
                trained_model=trained_model,
                decode=True,
            )

    def test_decode_image_no_trained_model(self, grayscale_image, grayscale_image_input_shape):
        """Test the decode_image method with no trained model"""
        with pytest.raises(ValueError):
            self.model(grayscale_image_input_shape, image=grayscale_image, decode=True)

    def test_decode_image_no_image(self, trained_model, input_shape):
        """Test the decode_image method with no image"""
        with pytest.raises(ValueError):
            self.model(input_shape, trained_model=trained_model, decode=True)

    def test_decode_image_colour_image(self, trained_model, colour_image, colour_image_input_shape):
        """Test the decode_image method with a colour image"""
        with pytest.raises(ValueError):
            self.model(
                colour_image_input_shape,
                image=colour_image,
                trained_model=trained_model,
                decode=True,
            )
