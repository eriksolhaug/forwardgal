import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from forwardgal.data import ab_zeropoint, pixel_scale_arcsec


def make_header(bunit="MJy/sr", pixar_sr=2.11539874851881e-14):
    hdr = fits.Header()
    hdr["BUNIT"] = bunit
    if pixar_sr is not None:
        hdr["PIXAR_SR"] = pixar_sr
    return hdr


def test_ab_zeropoint_nircam_value():
    # Value for the F115W i2d mosaic (0.03"/px), computed by hand.
    assert ab_zeropoint(make_header()) == pytest.approx(28.086585, abs=1e-5)


def test_ab_zeropoint_definition():
    # A source of 3631 Jy must have m_AB = 0: put it all in one pixel.
    pixar = 1e-14
    zp = ab_zeropoint(make_header(pixar_sr=pixar))
    pixel_value_mjysr = 3631 / (1e6 * pixar)  # MJy/sr that gives 3631 Jy
    assert -2.5 * np.log10(pixel_value_mjysr) + zp == pytest.approx(0.0, abs=1e-10)


def test_ab_zeropoint_wrong_unit():
    with pytest.raises(ValueError):
        ab_zeropoint(make_header(bunit="DN/s"))


def test_ab_zeropoint_missing_pixar():
    with pytest.raises(ValueError):
        ab_zeropoint(make_header(pixar_sr=None))


def test_pixel_scale():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]  # 0.03"/px, east left
    assert pixel_scale_arcsec(w) == pytest.approx(0.03)


def test_pixel_scale_non_square():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.03 / 3600, 0.04 / 3600]
    with pytest.raises(ValueError):
        pixel_scale_arcsec(w)
