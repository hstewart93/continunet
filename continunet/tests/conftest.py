import numpy as np
import pytest

from astropy.io import fits

from continunet.image.fits import ImageSquare


@pytest.fixture
def fits_file(tmp_path):
    """Fixture for creating a temporary FITS file with random data."""
    data = np.random.randint(0, 10, size=(1, 256, 256), dtype=np.uint8)
    hdu = fits.PrimaryHDU(data)
    header = fits.Header()

    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 3
    header["NAXIS1"] = 256
    header["NAXIS2"] = 256
    header["NAXIS3"] = 1
    header["BMAJ"] = 0.01
    header["BMIN"] = 0.01
    header["WCSAXES"] = 3
    header["CRPIX1"] = 128
    header["CRPIX2"] = 128
    header["CRPIX3"] = 1
    header["CDELT1"] = -0.0001
    header["CDELT2"] = 0.0001
    header["CDELT3"] = 1000
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT3"] = "Hz"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CTYPE3"] = "FREQ"
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CRVAL3"] = 9.5e09
    header["LONGPOLE"] = 180
    header["LATPOLE"] = 0
    header["RESTFRQ"] = 1.42040575200e09
    header["RADESYS"] = "ICRS"

    hdu.header = header
    path = tmp_path / "test.fits"
    hdu.writeto(path)

    yield path
    path.unlink()


@pytest.fixture
def trained_model():
    """Fixture for a trained model."""
    return "continunet/network/trained_model.h5"


@pytest.fixture
def grayscale_image():
    """Generate a random 256x256x1 image array."""
    image = np.random.randint(0, 10, size=(256, 256, 1), dtype=np.uint8)
    return image.reshape((1, 256, 256, 1))


@pytest.fixture
def colour_image():
    """Generate a random 256x256x3 image array."""
    image = np.random.randint(0, 10, size=(256, 256, 3), dtype=np.uint8)
    return image.reshape((1, 256, 256, 3))


@pytest.fixture
def invalid_image():
    """Generate an invalid shape image array, not divisble by 256."""
    image = np.random.randint(0, 10, size=(255, 255, 1), dtype=np.uint8)
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


@pytest.fixture
def valid_image_object(fits_file):
    """Fixture for a valid image object with data."""
    return ImageSquare(fits_file)


@pytest.fixture
def invalid_image_object(fits_file):
    """Fixture for image object with invalid image shape."""
    image = ImageSquare(fits_file)
    image.data = np.random.randint(0, 10, size=(1, 260, 260), dtype=np.uint8)
    return image


@pytest.fixture
def valid_image_shape():
    """Fixture for a valid image shape."""
    return (1, 256, 256, 1)


@pytest.fixture
def nan_image_object(fits_file):
    """Fixture for a valid image object witha NaN value."""
    image = ImageSquare(fits_file)
    image.data[0, 0, 0] = np.nan

    return image


@pytest.fixture
def image_object_all_nans(fits_file):
    """Fixture for a valid image object with all NaN values."""
    image = ImageSquare(fits_file)
    image.data = np.full_like(image.data, np.nan)

    return image
