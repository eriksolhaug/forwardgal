import numpy as np
import pytest
from scipy.optimize import brentq

import forwardgal  # noqa: F401
from forwardgal.images import find_images, triangle_candidates
from test_lenstronomy import cored_isothermal_field
from test_traced import linear_field

THETA_E, CORE = 10.0, 2.0


def axis_images(b):
    """Images of a source at (b, 0) for the cored isothermal lens, on the x axis.

    Solve x - theta_E x / sqrt(x^2 + c^2) = b for x (all roots).
    """
    f = lambda x: x - THETA_E * x / np.sqrt(x**2 + CORE**2) - b  # noqa: E731
    xs = np.linspace(-15, 15, 30001)
    v = f(xs)
    return sorted(brentq(f, xs[i], xs[i + 1]) for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0])


@pytest.mark.parametrize("interpolation", ["bicubic", "bilinear"])
def test_three_images_of_cored_isothermal_lens(interpolation):
    field = cored_isothermal_field(THETA_E, CORE, interpolation=interpolation)
    b = 1.3
    expected = axis_images(b)  # three images: outer, inner (other side), central
    assert len(expected) == 3
    found = find_images(field, b, 0.0)
    assert len(found) == 3
    xs = sorted(im.theta[0] for im in found)
    # The map is the exact field sampled at 0.126"; interpolation error dominates.
    tol = 2e-3 if interpolation == "bicubic" else 2e-2
    np.testing.assert_allclose(xs, expected, atol=tol)
    np.testing.assert_allclose([im.theta[1] for im in found], 0.0, atol=tol)
    assert all(im.residual < 1e-8 for im in found)


def test_magnifications_and_parity():
    field = cored_isothermal_field(THETA_E, CORE)
    found = find_images(field, 1.3, 0.0)
    # Analytic mu for an axisymmetric lens: 1 / [(1 - a/r)(1 - da/dr)].
    def mu(x):
        r = abs(x)
        a = THETA_E * r / np.sqrt(r**2 + CORE**2)
        da = THETA_E * CORE**2 / (r**2 + CORE**2) ** 1.5
        return 1.0 / ((1 - a / r) * (1 - da))
    for im in found:
        assert im.mu == pytest.approx(mu(im.theta[0]), rel=0.02)
    # outer image positive parity, the image inside the Einstein ring negative
    by_x = sorted(found, key=lambda im: im.theta[0])
    assert by_x[-1].mu > 0 and by_x[0].mu < 0


def test_single_image_for_linear_lens():
    field = linear_field(0.3, 0.1)  # beta = (0.6 theta_x, 0.8 theta_y)
    found = find_images(field, 1.2, -0.8)
    assert len(found) == 1
    assert found[0].theta == pytest.approx((2.0, -1.0), abs=1e-9)
    assert found[0].mu == pytest.approx(1 / (0.6 * 0.8), rel=1e-9)


def test_no_candidates_outside_source_plane_coverage():
    field = linear_field(0.0, 0.0)  # identity: map covers |theta| < 25"
    assert len(triangle_candidates(field, 100.0, 0.0)) == 0
    assert find_images(field, 100.0, 0.0) == []
