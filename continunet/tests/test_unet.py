"""Tests for the UNet model."""

from continunet.network.unet import Unet


class TestUnet:
    """Tests for the UNet model."""

    model = Unet

    def test_build_model(self):
        """Test the compile_model method"""

        model = self.model((256, 256, 1)).build_model()

        assert model is not None
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)
        assert len(model.layers) == 49

    def test_load_weights(self, trained_model):
        """Test the load_weights method"""

        model = self.model((256, 256, 1)).build_model()
        model.load_weights(trained_model)

        assert model is not None
        assert model.input_shape == (None, 256, 256, 1)
        assert model.output_shape == (None, 256, 256, 1)
        assert len(model.layers) == 49
        assert model.get_weights() is not None
