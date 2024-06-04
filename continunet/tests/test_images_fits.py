"""Tests for the images.fits module"""

import pytest

from continunet.image.fits import ImageSquare


class TestImageSquare:
    """Tests for the ImageSquare class"""

    model = ImageSquare

    def test_load(self, fits_file):
        """Test the load method"""

        image = self.model(fits_file)

        assert image.data is not None
        assert image.header is not None
        assert image.wcs is not None
        assert image.beam_size is not None
        assert image.shape is not None

        with pytest.raises(ValueError):
            self.model(None)

        assert image.header["NAXIS"] == 2
        assert image.header["NAXIS1"] == 100
        assert image.header["NAXIS2"] == 100
        assert image.header["BMAJ"] == 0.01
        assert image.header["BMIN"] == 0.01
        assert image.header["CDELT1"] == -0.0001
        assert image.header["CDELT2"] == 0.0001

        assert image.shape == (100, 100)

    def test_get_beam_size(self, fits_file):
        """Test the get_beam_size method"""

        image = self.model(fits_file)
        assert image.get_beam_size() == 0.01 * 3600

        del image.header["BMAJ"]
        with pytest.raises(KeyError):
            image.get_beam_size()

        del image.header["BMIN"]
        with pytest.raises(KeyError):
            image.get_beam_size()

        image.header["BMAJ"] = 0.01
        image.header["BMIN"] = 0.02
        with pytest.raises(KeyError):
            image.get_beam_size()
