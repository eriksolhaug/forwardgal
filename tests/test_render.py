import warnings

import numpy as np
import pytest

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.diagnostics import rendering_check
from forwardgal.profiles.sersic import sersic_polygon_flux, sersic_total_flux
from forwardgal.render import RenderSettings, render_sersic, render_sersic_exact

N_VALUES = [0.5, 0.62, 1.0, 1.37, 2.0, 2.9, 4.0, 5.3, 8.0]
Q, PA = 0.6, 30.0


def box_flux(npix, pixscale, x0, y0, I_e, R_e, n):
    """Exact flux inside the whole (square) image, as one big polygon.

    Additivity: the sum of exact pixel fluxes must equal this. It also takes
    care of the flux beyond the image edge, which is large for n = 8.
    """
    h = npix / 2 * pixscale
    box = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return float(sersic_polygon_flux(box, x0, y0, I_e, R_e, n, Q, PA))


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("re_pix", [1.0, 3.0, 10.0])
def test_hybrid_total_flux(n, re_pix):
    # Sum of hybrid pixel values x pixel area = exact flux in the image box.
    pixscale, npix = 0.03, 201
    x0, y0 = 0.13 * pixscale, -0.27 * pixscale  # off pixel centre
    R_e = re_pix * pixscale
    img = render_sersic((npix, npix), pixscale, npix // 2, npix // 2, x0, y0, 1.0, R_e, n, Q, PA)
    numeric = float(img.sum()) * pixscale**2
    assert numeric == pytest.approx(box_flux(npix, pixscale, x0, y0, 1.0, R_e, n), rel=1e-3)


def test_box_flux_equals_total_flux_for_compact_profile():
    # Sanity check of the reference: a compact n = 1 profile is fully inside.
    pixscale, R_e = 0.03, 0.03
    L = float(sersic_total_flux(1.0, R_e, 1.0, Q))
    assert box_flux(201, pixscale, 0.0, 0.0, 1.0, R_e, 1.0) == pytest.approx(L, rel=1e-10)


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0, 8.0])
@pytest.mark.parametrize("re_pix", [1.0, 3.0, 10.0])
def test_hybrid_matches_exact_per_pixel(n, re_pix):
    # Per pixel, relative to the peak pixel (what matters for chi^2 is the
    # absolute error, and pixels far in the tail are tiny).
    pixscale, npix = 0.03, 41
    x0, y0 = 0.13 * pixscale, -0.27 * pixscale
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=20, col0=20, x0=x0, y0=y0,
              I_e=1.0, R_e=re_pix * pixscale, n=n, q=Q, pa_deg=PA)
    hybrid = np.asarray(render_sersic(**kw))
    exact = np.asarray(render_sersic_exact(**kw))
    assert np.max(np.abs(hybrid - exact)) / exact.max() < 2e-3
    # Inside the 7 x 7 patch the two must agree to the exact method's accuracy.
    rc, cc = 20 + round(-0.27), 20 - round(0.13)  # pixel containing the centre
    patch = (slice(rc - 3, rc + 4), slice(cc - 3, cc + 4))
    np.testing.assert_allclose(hybrid[patch], exact[patch], rtol=1e-12)


def test_patch_shifted_inside_at_image_edge():
    # Centre in the corner pixel: the patch must stay inside the image.
    pixscale, npix = 0.03, 21
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=10, col0=10,
              x0=10 * pixscale, y0=-10 * pixscale,  # pixel [0, 0]
              I_e=1.0, R_e=0.03, n=4.0, q=Q, pa_deg=PA)
    hybrid = np.asarray(render_sersic(**kw))
    exact = np.asarray(render_sersic_exact(**kw))
    np.testing.assert_allclose(hybrid[:7, :7], exact[:7, :7], rtol=1e-12)


def test_render_settings_change_patch_size():
    pixscale, npix = 0.03, 21
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=10, col0=10, x0=0.0, y0=0.0,
              I_e=1.0, R_e=0.03, n=4.0, q=Q, pa_deg=PA)
    a = np.asarray(render_sersic(**kw, settings=RenderSettings(patch=3)))
    b = np.asarray(render_sersic(**kw, settings=RenderSettings(patch=7)))
    assert np.allclose(a[9:12, 9:12], b[9:12, 9:12], rtol=1e-12)
    assert not np.allclose(a[7, 7], b[7, 7], rtol=1e-12)  # outside 3x3, inside 7x7


# --- rendering check -----------------------------------------------------------


def test_rendering_check_passes_and_fails():
    rng = np.random.default_rng(1)
    exact = rng.uniform(1, 2, (10, 10))
    data = exact + rng.normal(0, 0.1, exact.shape)
    sigma = np.full(exact.shape, 0.1)
    mask = np.ones(exact.shape, bool)

    small = exact + 0.001  # 0.01 sigma everywhere
    res = rendering_check(small, exact, data, sigma, mask)
    assert res["ok"] and res["max_dev_sigma"] == pytest.approx(0.01)

    big = exact.copy()
    big[3, 3] += 0.05  # 0.5 sigma in one pixel
    with pytest.warns(UserWarning, match="rendering check failed"):
        res = rendering_check(big, exact, data, sigma, mask)
    assert not res["ok"] and res["max_dev_sigma"] == pytest.approx(0.5)


def test_rendering_check_uses_convolution_and_mask():
    exact = np.zeros((5, 5))
    hybrid = exact.copy()
    hybrid[0, 0] = 1.0  # large difference, but outside the mask
    mask = np.zeros((5, 5), bool)
    mask[2:, 2:] = True
    sigma = np.ones((5, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rendering_check(hybrid, exact, exact, sigma, mask)["ok"]
    # A "PSF" that moves flux from [0, 0] into the mask makes the check fail.
    spread = lambda im: im + im[0, 0]  # noqa: E731
    with pytest.warns(UserWarning):
        assert not rendering_check(hybrid, exact, exact, sigma, mask, convolve=spread)["ok"]


# --- Fourier modes (M12c) ----------------------------------------------------

MODES = ((1, 3, 4), (0.15, 0.05, 0.08), (70.0, -30.0, 25.0))


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0, 8.0])
@pytest.mark.parametrize("re_pix", [1.0, 3.0, 10.0])
def test_hybrid_matches_exact_per_pixel_with_modes(n, re_pix):
    """Same criterion as without modes: hybrid vs all-exact < 2e-3 of the peak."""
    pixscale, npix = 0.03, 41
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=20, col0=20, x0=0.13 * pixscale,
              y0=-0.27 * pixscale, I_e=1.0, R_e=re_pix * pixscale, n=n, q=Q, pa_deg=PA, modes=MODES)
    hybrid = np.asarray(render_sersic(**kw))
    exact = np.asarray(render_sersic_exact(**kw))
    assert np.max(np.abs(hybrid - exact)) / exact.max() < 2e-3


def test_modes_change_the_image_but_not_the_box_flux_formula():
    """Additivity with modes: the sum of exact pixel fluxes equals the flux of
    the whole box computed as one polygon with the same modes."""
    pixscale, npix = 0.03, 41
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=20, col0=20, x0=0.004, y0=-0.008,
              I_e=1.0, R_e=0.09, n=1.0, q=Q, pa_deg=PA)
    plain = np.asarray(render_sersic_exact(**kw))
    pert = np.asarray(render_sersic_exact(**kw, modes=MODES))
    assert np.max(np.abs(pert - plain)) / plain.max() > 0.05
    h = npix / 2 * pixscale
    box = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    whole = float(sersic_polygon_flux(box, 0.004, -0.008, 1.0, 0.09, 1.0, Q, PA, modes=MODES))
    assert pert.sum() * pixscale**2 == pytest.approx(whole, rel=1e-6)


# [M12e paused] # --- spiral rotation (M12e) -------------------------------------------------

# [M12e paused] from forwardgal.profiles.sersic import Spiral  # noqa: E402


# [M12e paused] # Realistic winding (90 deg over 1-8 px) at the default spiral_split = 4, and an
# [M12e paused] # extreme one (180 deg within 1-6 px) that needs spiral_split = 8 (measured
# [M12e paused] # 2026-11-16: 2.5e-3 of the peak at 4, 7.6e-4 at 8).
# [M12e paused] SPIRALS = [(Spiral("power", 0.03, 0.24, 90.0, 0.0), 4), (Spiral("power", 0.03, 0.18, 180.0, 0.0), 8)]


# [M12e paused] @pytest.mark.parametrize("n", [1.0, 4.0])
# [M12e paused] @pytest.mark.parametrize("sp, split", SPIRALS)
# [M12e paused] def test_hybrid_matches_fine_exact_with_spiral(n, sp, split):
    # [M12e paused] """Hybrid vs all-exact with spiral_split 16 (reference): < 2e-3 of the peak."""
    # [M12e paused] pixscale, npix = 0.03, 41
    # [M12e paused] kw = dict(shape=(npix, npix), pixscale=pixscale, row0=20, col0=20, x0=0.13 * pixscale,
              # [M12e paused] y0=-0.27 * pixscale, I_e=1.0, R_e=0.09, n=n, q=0.4, pa_deg=PA, modes=MODES, spiral=sp)
    # [M12e paused] hybrid = np.asarray(render_sersic(**kw, settings=RenderSettings(spiral_split=split)))
    # [M12e paused] ref = np.asarray(render_sersic_exact(**kw, spiral_split=16))
    # [M12e paused] assert np.max(np.abs(hybrid - ref)) / ref.max() < 2e-3


# --- bending modes (M12f) -----------------------------------------------------

from forwardgal.profiles.sersic import Bending  # noqa: E402


@pytest.mark.parametrize("n", [1.0, 4.0])
def test_hybrid_matches_fine_exact_with_bending(n):
    """Hybrid (default bend_split 4) vs all-exact with bend_split 16: < 2e-3 of the peak."""
    pixscale, npix = 0.03, 41
    b = Bending((2, 3), [0.02, -0.01], PA)  # ridge shifts of 0.2-0.3 R_e at R_e
    kw = dict(shape=(npix, npix), pixscale=pixscale, row0=20, col0=20, x0=0.13 * pixscale,
              y0=-0.27 * pixscale, I_e=1.0, R_e=0.09, n=n, q=0.4, pa_deg=PA, modes=MODES, bending=b)
    hybrid = np.asarray(render_sersic(**kw))
    ref = np.asarray(render_sersic_exact(**kw, bend_split=16))
    assert np.max(np.abs(hybrid - ref)) / ref.max() < 2e-3
