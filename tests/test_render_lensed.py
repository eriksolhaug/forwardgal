import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.profiles.sersic import sersic_total_flux
from forwardgal.render import (
    RenderSettings,
    _footprints,
    nearest_source_pixels,
    render_lensed_sersic,
    render_lensed_sersic_exact,
    render_sersic,
    render_sersic_exact,
)
from forwardgal.traced import pixel_theta, trace_grid
from test_traced import RA0, DEC0, image_wcs, linear_field  # synthetic lens helpers

PIX = 0.03
N = 41  # image is N x N pixels, centre pixel [20, 20]


def grid_and_centre(kappa, gamma, s=5, offset=(3.0, -2.0)):
    """Traced grid for a linear lens, and the beta of the image-centre pixel.

    ``offset``: image centre relative to the lens reference (arcsec).
    """
    field = linear_field(kappa, gamma)
    w = image_wcs(offset_arcsec=offset, pixscale=PIX)
    w.wcs.crpix = [21.0, 21.0]  # image centre pixel [20, 20] at crval
    grid = trace_grid(w, (N, N), field, subsample=s)
    tx, ty = pixel_theta(w, field, np.array(20.13), np.array(19.74))  # off pixel centre
    bx, by = field.backtrace(tx, ty)
    return grid, w, field, float(bx), float(by)


def test_no_lens_equals_unlensed_render():
    # Image tangent point = lens reference, so the image's north-up frame and
    # the lens tangent plane coincide. (With tangent points e.g. 3.6" apart the
    # frames are rotated by ~2e-6 rad, which local_theta ignores but the
    # lensed path, using the true WCS, includes; near a steep core that alone
    # changes pixel values by ~1e-4.)
    grid, w, field, bx, by = grid_and_centre(0.0, 0.0, offset=(0.0, 0.0))
    kw = dict(I_e=1.0, R_e=0.06, n=4.0, q=0.6, pa_deg=30.0)
    col0, row0 = w.wcs_world2pix(RA0, DEC0, 0)
    flat_kw = dict(shape=(N, N), pixscale=PIX, row0=float(row0), col0=float(col0), x0=bx, y0=by, **kw)
    # (a) Uniform parts agree away from both exact regions (> 9 px from centre).
    lensed = np.asarray(render_lensed_sersic(grid, bx, by, **kw))
    flat = np.asarray(render_sersic(**flat_kw))
    r, c = np.mgrid[0:N, 0:N]
    far = np.hypot(r - 20, c - 20) > 9
    np.testing.assert_allclose(lensed[far], flat[far], rtol=1e-8)
    # (b) Exact parts agree when every pixel is integrated exactly.
    lensed_all = np.asarray(render_lensed_sersic(grid, bx, by, **kw, settings=RenderSettings(exact_pixels=N * N)))
    np.testing.assert_allclose(lensed_all, np.asarray(render_sersic_exact(**flat_kw)), rtol=1e-8)


@pytest.mark.parametrize("kappa, gamma", [(0.3, 0.2), (0.2, 0.55), (0.5, 0.7)])
def test_magnification_theorem(kappa, gamma):
    # Total lensed flux = |mu| x source flux, mu = 1 / det A. (0.5, 0.7) is a
    # parity-flipped image (det < 0).
    grid, w, field, bx, by = grid_and_centre(kappa, gamma)
    R_e, n, q = 0.02, 1.0, 0.7
    img = np.asarray(render_lensed_sersic(grid, bx, by, 1.0, R_e, n, q, 40.0))
    lensed_flux = img.sum() * PIX**2
    mu = 1.0 / ((1 - kappa - gamma) * (1 - kappa + gamma))
    assert lensed_flux == pytest.approx(abs(mu) * float(sersic_total_flux(1.0, R_e, n, q)), rel=1e-3)


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0, 8.0])
def test_hybrid_matches_exact_lensed(n):
    grid, w, field, bx, by = grid_and_centre(0.2, 0.55)  # stretched 4.0x along one axis
    kw = dict(I_e=1.0, R_e=0.03, n=n, q=0.6, pa_deg=30.0)
    hybrid = np.asarray(render_lensed_sersic(grid, bx, by, **kw))
    exact = np.asarray(render_lensed_sersic_exact(grid, bx, by, **kw))
    assert np.max(np.abs(hybrid - exact)) / exact.max() < 2e-3


def test_exact_pixels_follow_the_arc():
    # Lens stretching 10x along theta_x (mu_t = 10, mu_r = 1): the pixels nearest
    # the centre in the source plane form an ellipse ~10x longer along x.
    grid, w, field, bx, by = grid_and_centre(0.45, 0.45)
    quads = _footprints(np.asarray(grid.beta_corner_x), np.asarray(grid.beta_corner_y))
    flat = np.asarray(nearest_source_pixels(quads, bx, by, 200))
    rows, cols = flat // N, flat % N
    assert (cols.max() - cols.min()) > 5 * (rows.max() - rows.min())
    assert (20, 20) in set(zip(rows.tolist(), cols.tolist()))  # includes the centre pixel


@pytest.mark.parametrize("n", [1.0, 4.0, 8.0])
def test_stretched_image_accuracy(n):
    # With the default N = 200 exact pixels, a 10x stretched compact source is
    # rendered to < 1e-4 of the peak (a 7 x 7 square patch gave up to 1.7e-3).
    grid, w, field, bx, by = grid_and_centre(0.45, 0.45)
    kw = dict(I_e=1.0, R_e=0.03, n=n, q=0.7, pa_deg=20.0)
    hybrid = np.asarray(render_lensed_sersic(grid, bx, by, **kw))
    exact = np.asarray(render_lensed_sersic_exact(grid, bx, by, **kw))
    assert np.max(np.abs(hybrid - exact)) / exact.max() < 1e-4


def test_subsample_mismatch_rejected():
    grid, *_ = grid_and_centre(0.0, 0.0, s=3)
    with pytest.raises(ValueError, match="subsample|s=3"):
        render_lensed_sersic(grid, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 0.0, settings=RenderSettings(subsample=5))


# --- image-plane (unlensed) components --------------------------------------------


@pytest.mark.parametrize("kappa, gamma", [(0.0, 0.0), (0.45, 0.45)])
def test_image_plane_component_ignores_the_lens(kappa, gamma):
    # An image-plane component must look the same whatever the lens does: the
    # lensed grid (any kappa, gamma) evaluated in the "image" plane equals the
    # no-lens grid evaluated in the "source" plane (where beta = theta).
    grid_lens, w, field, *_ = grid_and_centre(kappa, gamma, offset=(0.0, 0.0))
    grid_none, *_ = grid_and_centre(0.0, 0.0, offset=(0.0, 0.0))
    tx, ty = pixel_theta(w, field, np.array(19.6), np.array(21.3))
    kw = dict(x0=float(tx), y0=float(ty), I_e=1.0, R_e=0.05, n=4.0, q=0.7, pa_deg=-20.0)
    image_plane = np.asarray(render_lensed_sersic(grid_lens, **kw, plane="image"))
    reference = np.asarray(render_lensed_sersic(grid_none, **kw, plane="source"))
    np.testing.assert_allclose(image_plane, reference, rtol=1e-10, atol=1e-14)


def test_bad_plane_rejected():
    grid, *_ = grid_and_centre(0.0, 0.0)
    with pytest.raises(ValueError, match="plane"):
        render_lensed_sersic(grid, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 0.0, plane="lens")


def test_source_plane_image_orientation_and_flux():
    from forwardgal.render import source_plane_image
    comp = dict(x0=0.2, y0=-0.1, I_e=1.0, R_e=0.1, n=1.0, q=0.5, pa_deg=0.0)
    img, extent = source_plane_image([comp], centre=(0.2, -0.1), size=2.0, pixscale=0.01)
    assert img.shape == (200, 200)
    assert extent[0] > extent[1]  # east (larger beta_x) on the left
    # Elongated north-south (PA = 0): more extent along rows than columns.
    above = img > 0.5 * img.max()
    rows, cols = np.nonzero(above)
    assert np.ptp(rows) > 1.5 * np.ptp(cols)
    total = img.sum() * 0.01**2
    assert total == pytest.approx(float(sersic_total_flux(1.0, 0.1, 1.0, 0.5)), rel=2e-3)


# --- M9d: unit-flux templates -------------------------------------------------------

from forwardgal.profiles.sersic import unit_flux_ie  # noqa: E402


@pytest.mark.parametrize("kappa, gamma", [(0.0, 0.0), (0.3, 0.2)])
def test_unit_flux_template_sums_to_magnification(kappa, gamma):
    # Rendered with I_e = unit_flux_ie, a compact source's pixel values sum to
    # 1 x |mu|: the amplitude is then the INTRINSIC total flux.
    grid, w, field, bx, by = grid_and_centre(kappa, gamma)
    R_e, n, q = 0.02, 1.0, 0.7
    img = np.asarray(render_lensed_sersic(grid, bx, by, float(unit_flux_ie(R_e, n, q, PIX)), R_e, n, q, 40.0))
    mu = 1.0 / ((1 - kappa - gamma) * (1 - kappa + gamma))
    assert img.sum() == pytest.approx(abs(mu), rel=1e-3)


@pytest.mark.parametrize("n", [1.0, 4.0])
def test_hybrid_matches_exact_lensed_with_modes(n):
    grid, w, field, bx, by = grid_and_centre(0.2, 0.55)
    kw = dict(I_e=1.0, R_e=0.03, n=n, q=0.6, pa_deg=30.0, modes=((1, 3, 4), (0.15, 0.05, 0.08), (70.0, -30.0, 25.0)))
    hybrid = np.asarray(render_lensed_sersic(grid, bx, by, **kw))
    exact = np.asarray(render_lensed_sersic_exact(grid, bx, by, **kw))
    assert np.max(np.abs(hybrid - exact)) / exact.max() < 2e-3


# [M12e paused] @pytest.mark.parametrize("r_in, r_out, theta_out, split", [(0.03, 0.24, 90.0, 4), (0.01, 0.06, 150.0, 8)])
# [M12e paused] def test_hybrid_matches_fine_exact_lensed_with_spiral(r_in, r_out, theta_out, split):
    # [M12e paused] """Realistic winding at the default split 4; extreme (150 deg within 2 px)
    # [M12e paused] needs split 8 (measured: 4.9e-3 of the peak at 4, 9.2e-4 at 8)."""
    # [M12e paused] from forwardgal.profiles.sersic import Spiral

    # [M12e paused] grid, w, field, bx, by = grid_and_centre(0.2, 0.55)
    # [M12e paused] kw = dict(I_e=1.0, R_e=0.03, n=1.0, q=0.5, pa_deg=30.0, spiral=Spiral("power", r_in, r_out, theta_out, 0.0))
    # [M12e paused] hybrid = np.asarray(render_lensed_sersic(grid, bx, by, **kw, settings=RenderSettings(spiral_split=split)))
    # [M12e paused] ref = np.asarray(render_lensed_sersic_exact(grid, bx, by, **kw, spiral_split=16))
    # [M12e paused] assert np.max(np.abs(hybrid - ref)) / ref.max() < 2e-3


@pytest.mark.parametrize("b2, b3, split", [(0.003, -0.0015, 4), (0.008, -0.004, 8)])
def test_hybrid_matches_fine_exact_lensed_with_bending(b2, b3, split):
    """Mild bend (0.1 R_e at R_e) at the default split 4; a strong one (0.27 R_e
    at R_e, ~1 R_e at 2 R_e) needs split 8 (measured: 2.4e-3 of peak at 4, 4.9e-4 at 8)."""
    from forwardgal.profiles.sersic import Bending

    grid, w, field, bx, by = grid_and_centre(0.2, 0.55)
    kw = dict(I_e=1.0, R_e=0.03, n=1.0, q=0.5, pa_deg=30.0, bending=Bending((2, 3), [b2, b3], 30.0))
    hybrid = np.asarray(render_lensed_sersic(grid, bx, by, **kw, settings=RenderSettings(bend_split=split)))
    ref = np.asarray(render_lensed_sersic_exact(grid, bx, by, **kw, bend_split=16))
    assert np.max(np.abs(hybrid - ref)) / ref.max() < 2e-3
