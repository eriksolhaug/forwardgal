"""Independent cross-checks against lenstronomy (test-only dependency).

Conventions found when setting these up (documented in MILESTONES, M4c):
- lenstronomy INTERPOL wants increasing coordinate axes and f_x, f_y arrays
  indexed [y, x]; our theta_x decreases with column, so columns are reversed.
- lenstronomy SERSIC_ELLIPSE: R_sersic is the CIRCULARIZED radius
  (= our major-axis R_e * sqrt(q)); its angle phi is counter-clockwise from +x,
  so phi = 90 deg - PA (PA east of north, +x = east); amp = I_e.
- lenstronomy uses b_n ~ 1.9992 n - 0.3271 (3% off the exact value at
  n = 0.5); we use the exact b_n. The profile comparison below therefore
  evaluates OUR geometry with THEIR b_n, testing centre, q and PA handling.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

import forwardgal  # noqa: F401
from forwardgal.lens import DeflectionField
from forwardgal.profiles.sersic import elliptical_radius

lenstronomy = pytest.importorskip("lenstronomy")
from lenstronomy.LensModel.lens_model import LensModel  # noqa: E402
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil  # noqa: E402
from lenstronomy.LightModel.light_model import LightModel  # noqa: E402
from lenstronomy.Util.param_util import phi_q2_ellipticity  # noqa: E402

STEP = 0.12576


def cored_isothermal_field(theta_E=10.0, core=2.0, n=201, interpolation="bicubic"):
    """A smooth, NON-linear deflection field on a Lenstool-like grid."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [178.330251, 7.932518]
    w.wcs.crpix = [n // 2 + 1.0, n // 2 + 1.0]
    w.wcs.cdelt = [-STEP / 3600, STEP / 3600]
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    tx, ty = -STEP * (cols - n // 2), STEP * (rows - n // 2)
    r = np.sqrt(tx**2 + ty**2 + core**2)
    return DeflectionField(theta_E * tx / r, theta_E * ty / r, w, -STEP, STEP, n // 2, n // 2, 1.5,
                           interpolation=interpolation)


def to_lenstronomy(field):
    ny, nx = field.alpha_x.shape
    x_axis = field.step_x * (np.arange(nx) - field.j0)[::-1]  # increasing
    y_axis = field.step_y * (np.arange(ny) - field.i0)
    return [dict(grid_interp_x=x_axis, grid_interp_y=y_axis,
                 f_x=field.alpha_x[:, ::-1], f_y=field.alpha_y[:, ::-1])]


def test_backtrace_matches_lenstronomy_interpol():
    # lenstronomy INTERPOL interpolates bilinearly (spline kx = ky = 1).
    field = cored_isothermal_field(interpolation="bilinear")
    rng = np.random.default_rng(3)
    tx, ty = rng.uniform(-11, 11, 500), rng.uniform(-11, 11, 500)
    bx_l, by_l = LensModel(["INTERPOL"]).ray_shooting(tx, ty, to_lenstronomy(field))
    bx, by = field.backtrace(tx, ty)
    np.testing.assert_allclose(bx, bx_l, atol=1e-10)
    np.testing.assert_allclose(by, by_l, atol=1e-10)


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0])
@pytest.mark.parametrize("q, pa", [(1.0, 0.0), (0.5, 0.0), (0.5, 90.0), (0.3, 37.0)])
def test_sersic_geometry_matches_lenstronomy(n, q, pa):
    x0, y0, R_e, I_e = 0.05, -0.02, 0.2, 1.7
    e1, e2 = phi_q2_ellipticity(np.radians(90.0 - pa), q)
    kw = dict(amp=I_e, R_sersic=R_e * np.sqrt(q), n_sersic=n, e1=e1, e2=e2, center_x=x0, center_y=y0)
    rng = np.random.default_rng(4)
    x, y = rng.normal(x0, 0.3, 300), rng.normal(y0, 0.3, 300)
    theirs = LightModel(["SERSIC_ELLIPSE"]).surface_brightness(x, y, [kw])
    R = np.asarray(elliptical_radius(x, y, x0, y0, q, pa))
    ours_with_their_bn = I_e * np.exp(-SersicUtil.b_n(n) * ((R / R_e) ** (1 / n) - 1))
    np.testing.assert_allclose(ours_with_their_bn, theirs, rtol=1e-6)
