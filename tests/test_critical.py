import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.critical import caustics, critical_curves, jacobian_det
from test_lenstronomy import cored_isothermal_field
from test_traced import linear_field


@pytest.mark.parametrize("kappa, gamma", [(0.0, 0.0), (0.3, 0.2), (0.5, 0.7)])
def test_det_linear_lens(kappa, gamma):
    # alpha = ((kappa + gamma) x, (kappa - gamma) y): det A is constant. Checks
    # the column-direction sign (step_x < 0) in the x-derivative.
    det = jacobian_det(linear_field(kappa, gamma))
    np.testing.assert_allclose(det, (1 - kappa - gamma) * (1 - kappa + gamma), atol=1e-12)


def test_det_sign_of_x_derivative():
    # Shear along x only: alpha_x = g x  ->  det = 1 - g. If d/dtheta_x had the
    # wrong sign, det would come out as 1 + g.
    det = jacobian_det(linear_field(0.2, 0.2))  # alpha_x = 0.4 x, alpha_y = 0
    np.testing.assert_allclose(det, 0.6, atol=1e-12)


# Cored isothermal lens alpha(r) = theta_E r / sqrt(r^2 + c^2), theta_E = 10", c = 2":
#   tangential critical curve: alpha(r)/r = 1   -> r_t = sqrt(theta_E^2 - c^2)
#   radial critical curve:     d alpha/dr = 1   -> r_r = sqrt((theta_E c^2)^(2/3) - c^2)
THETA_E, CORE = 10.0, 2.0
R_T = np.sqrt(THETA_E**2 - CORE**2)
R_R = np.sqrt((THETA_E * CORE**2) ** (2 / 3) - CORE**2)


def radii(curves):
    return [np.hypot(c[:, 0], c[:, 1]) for c in curves]


def test_critical_curves_of_cored_isothermal_lens():
    field = cored_isothermal_field(THETA_E, CORE)
    rs = radii(critical_curves(field))
    assert len(rs) == 2
    tang = max(rs, key=np.mean)
    rad = min(rs, key=np.mean)
    # Finite differences + linear contour interpolation on a 0.126" grid.
    np.testing.assert_allclose(tang, R_T, atol=0.01)
    np.testing.assert_allclose(rad, R_R, atol=0.02)


def test_caustics_of_cored_isothermal_lens():
    field = cored_isothermal_field(THETA_E, CORE)
    curves = critical_curves(field)
    caus = caustics(field, curves)
    by_radius = sorted(zip(radii(curves), caus), key=lambda t: np.mean(t[0]))
    radial_caustic, tangential_caustic = by_radius[0][1], by_radius[1][1]
    # Circular lens: the tangential caustic degenerates to the point beta = 0.
    assert np.max(np.hypot(*tangential_caustic.T)) < 0.02
    # Radial caustic: circle of radius alpha(r_r) - r_r.
    r_caustic = THETA_E * R_R / np.sqrt(R_R**2 + CORE**2) - R_R
    np.testing.assert_allclose(np.hypot(*radial_caustic.T), r_caustic, atol=0.02)
