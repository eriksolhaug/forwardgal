import numpy as np
import pytest
from scipy.special import gammainc

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.grids import bin_mean, local_theta, subpixel_centres
from forwardgal.profiles.sersic import (
    b_n,
    fourier_angle,
    fourier_factor,
    fourier_flux_factor,
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


@pytest.mark.parametrize("n, re_pix", [(0.5, 3.0), (1.0, 3.0), (1.0, 10.0)])
def test_total_flux_uniform_sampling_smooth_profiles(n, re_pix):
    # Uniform sampling alone is fine for shallow profiles. (Steep profiles need
    # the exact patch: see tests/test_render.py.)
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


# --- M3b: exact pixel integration ---------------------------------------------

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.integrate import dblquad  # noqa: E402
from scipy.special import gammaincinv  # noqa: E402

from forwardgal.profiles.sersic import (  # noqa: E402
    sersic_polygon_flux,
    sersic_radial_integral,
)

# Sersic indices spanning the full allowed range, including non-round values.
N_VALUES = [0.5, 0.62, 0.8, 1.0, 1.37, 2.0, 2.9, 4.0, 5.3, 6.5, 8.0]
# Pixels (row, col) of a 1"/px grid: the one holding the centre, neighbours,
# a diagonal one, and two far out (to exercise the far-tail precision).
PIXELS = [(0, 0), (0, 1), (-1, 1), (2, -3), (5, 4)]
X0, Y0, Q, PA = -0.13, -0.27, 0.6, 30.0  # centre off pixel-centre inside (0, 0)


def test_radial_integral_limits():
    # G(0) = 0 and 2 pi q G(inf) = total flux.
    assert float(sersic_radial_integral(jnp.array(0.0), 1.0, 0.3, 4.0)) == 0.0
    G_inf = float(sersic_radial_integral(jnp.array(1e6), 2.0, 0.3, 4.0))
    assert 2 * np.pi * 0.6 * G_inf == pytest.approx(float(sersic_total_flux(2.0, 0.3, 4.0, 0.6)), rel=1e-10)


def numpy_sersic(x, y, R_e, n):
    """Plain-numpy Sersic (I_e = 1) at the test geometry, for scipy's dblquad."""
    pa = np.radians(PA)
    dx, dy = x - X0, y - Y0
    u = dx * np.sin(pa) + dy * np.cos(pa)
    v = dx * np.cos(pa) - dy * np.sin(pa)
    R = np.sqrt(u * u + (v / Q) ** 2)
    return np.exp(-gammaincinv(2 * n, 0.5) * ((R / R_e) ** (1 / n) - 1))


def pixel_bounds(row, col):
    """theta ranges of pixel (row, col) on a 1"/px north-up grid (east = -col)."""
    return -(col + 0.5), -(col - 0.5), row - 0.5, row + 0.5


def reference_pixel_flux(row, col, R_e, n):
    """Brute-force adaptive 2-D integral; split at the centre so the cusp is on a corner."""
    xa, xb, ya, yb = pixel_bounds(row, col)
    xs = sorted({xa, xb} | ({X0} if xa < X0 < xb else set()))
    ys = sorted({ya, yb} | ({Y0} if ya < Y0 < yb else set()))
    return sum(
        dblquad(lambda y, x: numpy_sersic(x, y, R_e, n), a, b, c, d, epsabs=0, epsrel=1e-11)[0]
        for a, b in zip(xs[:-1], xs[1:])
        for c, d in zip(ys[:-1], ys[1:])
    )


def pixel_vertices(row, col):
    xa, xb, ya, yb = pixel_bounds(row, col)
    return jnp.array([[xb, ya], [xa, ya], [xa, yb], [xb, yb]])


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("row, col", PIXELS)
def test_polygon_flux_matches_brute_force(n, row, col):
    # R_e = 1 pixel: the most compact (hardest) case.
    exact = float(sersic_polygon_flux(pixel_vertices(row, col), X0, Y0, 1.0, 1.0, n, Q, PA))
    ref = reference_pixel_flux(row, col, 1.0, n)
    assert exact == pytest.approx(ref, rel=2e-6)


def test_polygon_flux_gradients_finite_at_centre_pixel():
    verts = pixel_vertices(0, 0)
    f = lambda p: sersic_polygon_flux(verts, p[0], p[1], p[2], p[3], p[4], p[5], p[6])  # noqa: E731
    for n in (0.5, 1.0, 4.0, 8.0):
        g = jax.grad(f)(jnp.array([X0, Y0, 1.0, 1.0, n, Q, PA]))
        assert np.all(np.isfinite(g)), n


# --- Fourier modes (M12a) ---------------------------------------------------

MODES = ((1, 3, 4), (0.15, 0.05, 0.08), (70.0, -30.0, 25.0))  # (m, a_m, phi_m in degrees)
GEOM = dict(x0=0.3, y0=-0.2, I_e=1.0, R_e=0.5, n=1.5, q=0.6, pa_deg=-40.0)


def test_zero_amplitude_modes_change_nothing():
    x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    zero = ((1, 3, 4), (0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    np.testing.assert_allclose(sersic(x, y, **GEOM, modes=zero), sersic(x, y, **GEOM), rtol=1e-14)


def test_isophote_follows_input_modes():
    """Along each ray from the centre, the I = I_e isophote lies where
    R(point) * g(theta_G) = R_e. Find it by bisection on the rendered profile
    and compare with the prediction from the input (a_m, phi_m)."""
    g0 = GEOM
    for ang in np.linspace(0, 2 * np.pi, 37)[:-1]:
        d = np.array([np.cos(ang), np.sin(ang)])
        def f(t):
            return float(sersic(g0["x0"] + t * d[0], g0["y0"] + t * d[1], **g0, modes=MODES)) - g0["I_e"]
        lo, hi = 1e-3, 5.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        # prediction: R scales linearly with t along a ray and theta_G is constant
        px, py = g0["x0"] + d[0], g0["y0"] + d[1]
        R1 = np.hypot(*_uv(px, py))
        theta = float(fourier_angle(px, py, g0["x0"], g0["y0"], g0["q"], g0["pa_deg"]))
        t_pred = g0["R_e"] / (R1 * float(fourier_factor(theta, MODES)))
        assert lo == pytest.approx(t_pred, rel=1e-10), np.rad2deg(ang)


def _uv(px, py):
    """(u, v / q) of a point, for GEOM (helper for the isophote test)."""
    pa = np.deg2rad(GEOM["pa_deg"])
    dx, dy = px - GEOM["x0"], py - GEOM["y0"]
    return dx * np.sin(pa) + dy * np.cos(pa), (dx * np.cos(pa) - dy * np.sin(pa)) / GEOM["q"]


def test_m1_phase_zero_extends_toward_pa():
    """Regression test of the GALFIT convention measured on 2026-11-16: with
    a_1 > 0 and phi_1 = 0, g = 1 - a_1 along +PA, so the profile is brighter
    (more extended) on the +PA side."""
    pa = np.deg2rad(GEOM["pa_deg"])
    d = 0.4 * np.array([np.sin(pa), np.cos(pa)])  # along +PA (east, north components)
    m1 = ((1,), (0.2,), (0.0,))
    fwd = float(sersic(GEOM["x0"] + d[0], GEOM["y0"] + d[1], **GEOM, modes=m1))
    back = float(sersic(GEOM["x0"] - d[0], GEOM["y0"] - d[1], **GEOM, modes=m1))
    assert fwd > back


# --- Fourier-mode flux normalization (M12b) --------------------------------


def test_flux_factor_without_modes_is_one():
    assert fourier_flux_factor(None) == 1.0
    assert float(fourier_flux_factor(((1, 3, 4), (0.0, 0.0, 0.0), (5.0, 6.0, 7.0)))) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("m", [1, 3, 4])
@pytest.mark.parametrize("a", [0.05, 0.3, 0.7])
def test_flux_factor_single_mode_exact(m, a):
    """One mode: (1 / 2 pi) int (1 + a cos m theta)^-2 = (1 - a^2)^(-3/2), for any phase."""
    got = float(fourier_flux_factor(((m,), (a,), (23.0,))))
    assert got == pytest.approx((1 - a**2) ** -1.5, rel=1e-12)


def test_flux_factor_matches_brute_force_image_sum():
    """Sum of a finely sampled perturbed Sersic (n = 1, wide box) equals
    sersic_total_flux * fourier_flux_factor."""
    I_e, R_e, n, q, pa = 1.0, 0.3, 1.0, 0.6, 25.0
    h = 0.01
    x, y = np.meshgrid(np.arange(-6, 6, h) + h / 2, np.arange(-6, 6, h) + h / 2)
    img = np.asarray(sersic(x, y, 0.0, 0.0, I_e, R_e, n, q, pa, modes=MODES))
    expected = float(sersic_total_flux(I_e, R_e, n, q)) * float(fourier_flux_factor(MODES))
    assert img.sum() * h**2 == pytest.approx(expected, rel=1e-4)


def test_flux_factor_is_differentiable():
    import jax

    def f(a1):
        return fourier_flux_factor(((1, 4), (a1, 0.1), (30.0, 10.0)))

    g = float(jax.grad(f)(0.2))
    fd = (float(f(0.2 + 1e-6)) - float(f(0.2 - 1e-6))) / 2e-6
    assert g == pytest.approx(fd, rel=1e-6)


# --- Fourier modes in the exact pixel integral (M12c) ----------------------


def numpy_sersic_modes(x, y, R_e, n, modes):
    """Plain-numpy Sersic with Fourier modes (I_e = 1), test geometry; independent
    of the JAX code: theta_G = pi - atan2(v / q, u) (GALFIT, measured M12a)."""
    pa = np.radians(PA)
    dx, dy = x - X0, y - Y0
    u = dx * np.sin(pa) + dy * np.cos(pa)
    v = dx * np.cos(pa) - dy * np.sin(pa)
    th = np.pi - np.arctan2(v / Q, u)
    g = 1 + sum(a * np.cos(m * (th + np.radians(p))) for m, a, p in zip(*modes))
    R = np.sqrt(u * u + (v / Q) ** 2) * g
    return np.exp(-gammaincinv(2 * n, 0.5) * ((R / R_e) ** (1 / n) - 1))


def reference_pixel_flux_modes(row, col, R_e, n, modes):
    xa, xb, ya, yb = pixel_bounds(row, col)
    xs = sorted({xa, xb} | ({X0} if xa < X0 < xb else set()))
    ys = sorted({ya, yb} | ({Y0} if ya < Y0 < yb else set()))
    return sum(
        dblquad(lambda y, x: numpy_sersic_modes(x, y, R_e, n, modes), a, b, c, d, epsabs=0, epsrel=1e-10)[0]
        for a, b in zip(xs[:-1], xs[1:])
        for c, d in zip(ys[:-1], ys[1:])
    )


@pytest.mark.parametrize("n", [1.0, 4.0])
@pytest.mark.parametrize("row, col", [(0, 0), (1, 0), (-1, 2)])
def test_polygon_flux_with_modes_matches_brute_force(n, row, col):
    """Exact pixel flux with m = 1, 3, 4 modes vs scipy's adaptive 2-D integral
    (R_e = 1 pixel, the hardest case; (0, 0) contains the centre)."""
    exact = float(sersic_polygon_flux(pixel_vertices(row, col), X0, Y0, 1.0, 1.0, n, Q, PA, modes=MODES))
    ref = reference_pixel_flux_modes(row, col, 1.0, n, MODES)
    assert exact == pytest.approx(ref, rel=2e-6)


def test_polygon_flux_with_modes_gradients_finite_at_centre_pixel():
    verts = pixel_vertices(0, 0)

    def f(a):
        return sersic_polygon_flux(verts, X0, Y0, 1.0, 1.0, 4.0, Q, PA, modes=((1, 3, 4), a, (70.0, -30.0, 25.0)))

    g = jax.grad(f)(jnp.array([0.15, 0.05, 0.08]))
    assert np.all(np.isfinite(g))


# [M12e paused] # --- spiral rotation in the exact pixel integral (M12e) ---------------------

# [M12e paused] from forwardgal.profiles.sersic import Spiral, spiral_rotation  # noqa: E402

# [M12e paused] STRONG = Spiral("power", 1.0, 6.0, 180.0, 0.0)  # 180 deg of winding between 1 and 6 pixels


# [M12e paused] def numpy_sersic_spiral(x, y, R_e, n, sp):
    # [M12e paused] """Plain-numpy spiral Sersic (I_e = 1) at the test geometry, PA turning with r."""
    # [M12e paused] dx, dy = x - X0, y - Y0
    # [M12e paused] p = np.radians(PA + np.asarray(spiral_rotation(jnp.asarray(np.hypot(dx, dy)), sp)))
    # [M12e paused] u = dx * np.sin(p) + dy * np.cos(p)
    # [M12e paused] v = dx * np.cos(p) - dy * np.sin(p)
    # [M12e paused] return np.exp(-gammaincinv(2 * n, 0.5) * ((np.hypot(u, v / Q) / R_e) ** (1 / n) - 1))


# [M12e paused] def reference_pixel_flux_spiral(row, col, R_e, n, sp):
    # [M12e paused] xa, xb, ya, yb = pixel_bounds(row, col)
    # [M12e paused] xs = sorted({xa, xb} | ({X0} if xa < X0 < xb else set()))
    # [M12e paused] ys = sorted({ya, yb} | ({Y0} if ya < Y0 < yb else set()))
    # [M12e paused] return sum(
        # [M12e paused] dblquad(lambda y, x: numpy_sersic_spiral(x, y, R_e, n, sp), a, b, c, d, epsabs=0, epsrel=1e-9)[0]
        # [M12e paused] for a, b in zip(xs[:-1], xs[1:])
        # [M12e paused] for c, d in zip(ys[:-1], ys[1:])
    # [M12e paused] )


# [M12e paused] @pytest.mark.parametrize("split, tol", [(4, 5e-3), (8, 1.5e-3)])
# [M12e paused] @pytest.mark.parametrize("row, col", [(0, 0), (2, -1), (-4, 2)])
# [M12e paused] def test_spiral_exact_pixels_match_brute_force(split, tol, row, col):
    # [M12e paused] """Frozen rotation in split x split sub-quads: error ~1/split^2 (measured 2026-11-16:
    # [M12e paused] <= 3.6e-3 at split 4, <= 9e-4 at split 8 for this strong winding)."""
    # [M12e paused] exact = float(sersic_polygon_flux(pixel_vertices(row, col), X0, Y0, 1.0, 2.0, 4.0, Q, PA,
                                      # [M12e paused] spiral=STRONG, spiral_split=split))
    # [M12e paused] assert exact == pytest.approx(reference_pixel_flux_spiral(row, col, 2.0, 4.0, STRONG), rel=tol)


# [M12e paused] def test_zero_rotation_equals_no_spiral_and_has_finite_gradient():
    # [M12e paused] x, y = np.meshgrid(np.linspace(-3, 3, 31), np.linspace(-3, 3, 31))
    # [M12e paused] sp0 = Spiral("power", 0.5, 2.0, 0.0, 0.0)
    # [M12e paused] np.testing.assert_allclose(sersic(x, y, **GEOM, spiral=sp0), sersic(x, y, **GEOM), rtol=1e-12)

    # [M12e paused] def f(th):
        # [M12e paused] return jnp.sum(sersic(x, y, **GEOM, spiral=Spiral("power", 0.5, 2.0, th, 0.0)))

    # [M12e paused] assert np.isfinite(float(jax.grad(f)(0.0)))


# --- bending modes in the exact pixel integral (M12f) -----------------------

from forwardgal.profiles.sersic import Bending  # noqa: E402

STRONG_BEND = Bending((2, 3), [1.0, -0.6], 0.0)  # ~0.5 R_e bends with R_e = 2 pixels


@pytest.mark.parametrize("split, tol", [(4, 6e-3), (8, 1.5e-3)])
@pytest.mark.parametrize("row, col", [(0, 0), (2, -1), (-4, 2)])
def test_bent_exact_pixels_match_brute_force(split, tol, row, col):
    """Tangent-line shear in split x split sub-quads: error ~1/split^2
    (measured 2026-11-16 for this strong bend: <= 4.9e-3 at 4, <= 1.2e-3 at 8)."""
    xa, xb, ya, yb = pixel_bounds(row, col)
    xs = sorted({xa, xb} | ({X0} if xa < X0 < xb else set()))
    ys = sorted({ya, yb} | ({Y0} if ya < Y0 < yb else set()))

    def f(y, x):
        return float(sersic(x, y, X0, Y0, 1.0, 2.0, 4.0, Q, PA, bending=STRONG_BEND))

    ref = sum(dblquad(f, a, b, c, d, epsabs=0, epsrel=1e-9)[0]
              for a, b in zip(xs[:-1], xs[1:]) for c, d in zip(ys[:-1], ys[1:]))
    exact = float(sersic_polygon_flux(pixel_vertices(row, col), X0, Y0, 1.0, 2.0, 4.0, Q, PA,
                                      bending=STRONG_BEND, bend_split=split))
    assert exact == pytest.approx(ref, rel=tol)



def test_forward_derivatives_finite_at_n_half_on_rays_from_the_centre():
    """Regression (2026-11-16): at exactly n = 0.5 a ray starting at the centre
    (X_in = 0) gave NaN forward-mode derivatives (0^0 inside JAX's gamma
    derivative), which turned the whole LM Jacobian NaN when a fit hit n = 0.5."""
    from forwardgal.profiles.sersic import sersic_radial_integral_between

    X_in, X_out = jnp.array([0.0, 0.0, 0.1]), jnp.array([0.05, 2.0, 0.2])
    for n in (0.5, 1.0, 4.0):
        d = jax.jacfwd(lambda nn: sersic_radial_integral_between(X_in, X_out, 1.0, 0.39, nn))(n)
        assert np.all(np.isfinite(np.asarray(d))), n
    verts = pixel_vertices(0, 0)  # contains the centre
    J = jax.jacfwd(lambda p: sersic_polygon_flux(verts, p[0], p[1], 1.0, p[2], p[3], Q, PA))(
        jnp.array([X0, Y0, 1.0, 0.5]))
    assert np.all(np.isfinite(np.asarray(J)))
