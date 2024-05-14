import numpy as np
import pytest

from astropy.io import fits


@pytest.fixture
def fits_file(tmp_path):
    """Fixture for creating a temporary FITS file with random data."""
    data = np.random.rand(100, 100)

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 100
    header["NAXIS2"] = 100
    header["BMAJ"] = 0.01
    header["BMIN"] = 0.01
    header["CDELT1"] = -0.0001
    header["CDELT2"] = 0.0001

    path = tmp_path / "test.fits"
    fits.writeto(path, data, header)

    yield path
    path.unlink()


@pytest.fixture
def trained_model():
    """Fixture for a trained model."""
    return "continunet/network/trained_model.h5"


@pytest.fixture
def grayscale_image():
    """Generate a random 256x256x1 image array."""
    image = np.random.randint(0, 1, size=(256, 256, 1), dtype=np.uint8)
    return image.reshape((1, 256, 256, 1))


@pytest.fixture
def colour_image():
    """Generate a random 256x256x3 image array."""
    image = np.random.randint(0, 1, size=(256, 256, 3), dtype=np.uint8)
    return image.reshape((1, 256, 256, 3))


@pytest.fixture
def invalid_image():
    """Generate an invalid shape image array, not divisble by 256."""
    image = np.random.randint(0, 1, size=(255, 255, 1), dtype=np.uint8)
    return image.reshape((1, 255, 255, 1))


@pytest.fixture
def input_shape():
    """Fixture for the input shape."""
    return (256, 256, 1)


@pytest.fixture
def grayscale_image_input_shape(grayscale_image):
    """Fixture for the input shape."""
    return grayscale_image.shape[1:]


@pytest.fixture
def colour_image_input_shape(colour_image):
    """Fixture for the input shape."""
    return colour_image.shape[1:]


@pytest.fixture
def invalid_image_input_shape(invalid_image):
    """Fixture for the input shape."""
    return invalid_image.shape[1:]
