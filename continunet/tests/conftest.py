import numpy as np
import pytest

from astropy.io import fits


@pytest.fixture
def fits_file(tmp_path):
    """"""
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
