"""Tests for the UNet model."""

import numpy as np
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
            colour_image_input_shape,
            image=colour_image,
            trained_model=TRAINED_MODEL,
        )
        with pytest.raises(ValueError):
            test_model.decode_image()

    @pytest.mark.parametrize("size", [512, 1024])
    def test_large_input_shapes(self, size):
        """
        Test the UNet build and forward pass on larger image sizes
        to ensure shape concatenation works (no off-by-one mismatches).
        """

        input_shape = (size, size, 1)
        input_image = np.random.rand(1, *input_shape).astype(np.float32)
        print(f"\n[DEBUG] Testing UNet with input shape: {input_shape}")

        # Build model
        unet = self.model(input_shape)
        assert unet.model.input_shape == (None, *input_shape)

        # Forward pass: should run without shape mismatch
        try:
            output = unet.model.predict(input_image, verbose=0)
        except ValueError as e:
            pytest.fail(f"Shape mismatch error for input {input_shape}: {e}")

        # Validate output shape
        assert output.shape == (
            1,
            *input_shape,
        ), f"Output shape {output.shape} != expected {(1, *input_shape)}"
