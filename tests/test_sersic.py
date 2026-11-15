import numpy as np
import pytest
from scipy.special import gammainc

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.grids import bin_mean, local_theta, subpixel_centres
from forwardgal.profiles.sersic import (
    b_n,
    ie_to_mag,
    mag_to_ie,
    sersic,
    sersic_total_flux,
)

# --- grids ------------------------------------------------------------------


def test_subpixel_centres():
    rows, cols = subpixel_centres(2, 3, s=2)
    assert rows.shape == cols.shape == (4, 6)
    np.testing.assert_allclose(cols[0], [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25])
    np.testing.assert_allclose(rows[:, 0], [-0.25, 0.25, 0.75, 1.25])
    # s = 1 gives the data-pixel centres themselves.
    rows1, cols1 = subpixel_centres(2, 3, s=1)
    np.testing.assert_array_equal(cols1[0], [0, 1, 2])


def test_bin_mean():
    img = np.arange(16.0).reshape(4, 4)
    np.testing.assert_allclose(bin_mean(img, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_local_theta_orientation():
    # One column right = west (theta_x decreases); one row up = north.
    tx, ty = local_theta(np.array([5, 5, 6]), np.array([5, 6, 5]), 0.03, 5, 5)
    np.testing.assert_allclose(tx, [0, -0.03, 0])
    np.testing.assert_allclose(ty, [0, 0, 0.03])


# --- profile shape and conventions -------------------------------------------


@pytest.mark.parametrize("n", [0.3, 0.5, 0.6, 1.0, 2.0, 4.0, 8.0])
def test_b_n_half_light(n):
    # Definition: the regularized lower incomplete gamma P(2n, b_n) = 1/2.
    assert gammainc(2 * n, float(b_n(n))) == pytest.approx(0.5, abs=1e-12)


def test_b_n_is_differentiable():
    # Needed later for gradients with respect to n. Check against finite differences.
    import jax

    g = float(jax.grad(b_n)(1.5))
    fd = (float(b_n(1.5 + 1e-6)) - float(b_n(1.5 - 1e-6))) / 2e-6
    assert g == pytest.approx(fd, rel=1e-6)


def test_ie_at_re_along_major_axis():
    # PA = 0: major axis points north, so (0, R_e) is at R = R_e.
    v = sersic(0.0, 0.3, 0.0, 0.0, I_e=2.0, R_e=0.3, n=1.5, q=0.5, pa_deg=0.0)
    assert float(v) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("pa, major_dir", [(0.0, (0, 1)), (90.0, (1, 0)), (45.0, (1, 1))])
def test_pa_east_of_north(pa, major_dir):
    # A point along the major axis is brighter than one equally far along the minor axis.
    d = np.array(major_dir, float) / np.hypot(*major_dir) * 0.2
    minor = np.array([d[1], -d[0]])
    kw = dict(x0=0.0, y0=0.0, I_e=1.0, R_e=0.2, n=1.0, q=0.4, pa_deg=pa)
    on_major = float(sersic(d[0], d[1], **kw))
    on_minor = float(sersic(minor[0], minor[1], **kw))
    assert on_major == pytest.approx(1.0, rel=1e-12)  # R = R_e exactly
    assert on_minor < on_major


def test_circular_when_q_is_one():
    kw = dict(x0=0.1, y0=-0.2, I_e=1.0, R_e=0.3, n=2.0, q=1.0, pa_deg=37.0)
    ang = np.linspace(0, 2 * np.pi, 7)
    vals = sersic(0.1 + 0.25 * np.cos(ang), -0.2 + 0.25 * np.sin(ang), **kw)
    np.testing.assert_allclose(vals, vals[0], rtol=1e-12)


# --- total flux (needs Erik's sersic_total_flux) ------------------------------


def test_total_flux_exponential():
    # n = 1, Gamma(2) = 1: L = 2 pi q R_e^2 I_e e^b / b^2.
    b = float(b_n(1.0))
    expected = 2 * np.pi * 0.6 * 0.3**2 * 2.0 * np.exp(b) / b**2
    assert float(sersic_total_flux(2.0, 0.3, 1.0, 0.6)) == pytest.approx(expected, rel=1e-12)


CUSP = pytest.mark.xfail(
    strict=True,
    reason="known limitation: point-sampling the steep n=4 core over-weights the "
    "centre (+67% at R_e = 3 px, s = 5); needs central oversampling (see MILESTONES)",
)


@pytest.mark.parametrize(
    "n, re_pix",
    [(0.5, 3.0), (1.0, 3.0), (1.0, 10.0), pytest.param(4.0, 3.0, marks=CUSP)],
)
def test_total_flux_matches_rendered_image(n, re_pix):
    # Render on a large, well-subsampled grid and compare sum * pixel area to L.
    pixscale, npix, s = 0.03, 401, 5
    rows, cols = subpixel_centres(npix, npix, s)
    tx, ty = local_theta(rows, cols, pixscale, npix // 2, npix // 2)
    R_e, q, I_e = re_pix * pixscale, 0.6, 1.0
    img = bin_mean(sersic(tx, ty, 0.0, 0.0, I_e, R_e, n, q, 30.0), s)
    numeric = float(img.sum()) * pixscale**2
    assert numeric == pytest.approx(float(sersic_total_flux(I_e, R_e, n, q)), rel=1e-3)


def test_magnitude_roundtrip():
    kw = dict(R_e=0.1, n=2.5, q=0.7, zeropoint=28.087, pixscale=0.03)
    I_e = mag_to_ie(22.3, **kw)
    assert float(ie_to_mag(I_e, **kw)) == pytest.approx(22.3, abs=1e-10)
