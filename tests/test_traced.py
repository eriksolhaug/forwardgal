import numpy as np
import pytest
from astropy.wcs import WCS

import forwardgal  # noqa: F401
from forwardgal.lens import DeflectionField
from forwardgal.traced import pixel_theta, trace_grid

RA0, DEC0 = 178.330251, 7.932518
STEP = 0.12576  # arcsec per deflection-map pixel


def linear_field(kappa=0.0, gamma=0.0):
    """Deflection field alpha = (kappa + gamma) theta_x, (kappa - gamma) theta_y.

    Linear in theta, so bilinear interpolation is exact and
    det(d beta / d theta) = (1 - kappa - gamma) (1 - kappa + gamma).
    """
    n = 401
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [201.0, 201.0]
    w.wcs.cdelt = [-STEP / 3600, STEP / 3600]
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    tx, ty = -STEP * (cols - 200), STEP * (rows - 200)
    return DeflectionField(
        alpha_x=(kappa + gamma) * tx, alpha_y=(kappa - gamma) * ty, wcs=w,
        step_x=-STEP, step_y=STEP, i0=200.0, j0=200.0, z_ref=1.5,
    )


def image_wcs(offset_arcsec=(3.0, -2.0), pixscale=0.03):
    """North-up image WCS centred a few arcsec from the lens reference."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0 + offset_arcsec[0] / 3600 / np.cos(np.radians(DEC0)), DEC0 + offset_arcsec[1] / 3600]
    w.wcs.crpix = [11.0, 11.0]
    w.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    return w


@pytest.mark.parametrize("kappa, gamma", [(0.0, 0.0), (0.3, 0.0), (0.2, 0.15), (0.5, 0.7)])
def test_area_ratio_equals_det_A(kappa, gamma):
    # (0.5, 0.7) has det < 0: a parity-flipped (mirror) image region.
    g = trace_grid(image_wcs(), (20, 20), linear_field(kappa, gamma), subsample=3)
    det = (1 - kappa - gamma) * (1 - kappa + gamma)
    np.testing.assert_allclose(g.area_ratio(), det, rtol=1e-6)


def test_subpixels_backtraced():
    field = linear_field(0.2, 0.1)
    w = image_wcs()
    g = trace_grid(w, (4, 5), field, subsample=3, origin=(2, 7))
    # Subpixel 4 (the middle of 3 x 3) of pixel [1, 2] is the centre of image
    # pixel [2 + 1, 7 + 2].
    tx, ty = pixel_theta(w, field, np.array(3.0), np.array(9.0))
    assert g.beta_sub_x[1, 2, 4] == pytest.approx(float(tx) * (1 - 0.3), abs=1e-9)
    assert g.beta_sub_y[1, 2, 4] == pytest.approx(float(ty) * (1 - 0.1), abs=1e-9)


def test_corners_shared_and_shaped():
    g = trace_grid(image_wcs(), (6, 7), linear_field(0.1), subsample=1)
    assert g.beta_corner_x.shape == (7, 8)
    # The footprints of pixels [2, 3] and [2, 4] share an edge.
    a, b = g.footprint(2, 3), g.footprint(2, 4)
    np.testing.assert_array_equal(a[1], b[0])
    np.testing.assert_array_equal(a[2], b[3])


def test_outside_map_rejected():
    far = image_wcs(offset_arcsec=(30.0, 0.0))  # map covers only +-25"
    with pytest.raises(ValueError, match="outside the deflection maps"):
        trace_grid(far, (5, 5), linear_field(), subsample=1)
