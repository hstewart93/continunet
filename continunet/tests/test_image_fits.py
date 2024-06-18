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

        assert image.header["NAXIS"] == 3
        assert image.header["NAXIS1"] == 256
        assert image.header["NAXIS2"] == 256
        assert image.header["NAXIS3"] == 1
        assert image.header["BMAJ"] == 0.01
        assert image.header["BMIN"] == 0.01
        assert image.header["CDELT1"] == -0.0001
        assert image.header["CDELT2"] == 0.0001

        assert image.shape == (1, 256, 256)

    def test_load_no_celestial(self, fits_file_no_celestial):
        """Test the check_header method with no celestial information"""
        with pytest.raises(ValueError):
            self.model(fits_file_no_celestial)

    @pytest.mark.parametrize("key", ["CRPIX1", "CRPIX2"])
    def test_check_header(self, fits_file, key):
        """Test the check_header method"""

        image = self.model(fits_file)

        del image.header[key]
        with pytest.raises(KeyError):
            image.check_header()

    def test_wcs(self, fits_file):
        """Test the wcs property"""

        image = self.model(fits_file)
        assert image.wcs.has_celestial

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
