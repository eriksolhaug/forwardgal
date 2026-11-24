import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.quasar import image_fixed_quasar
from forwardgal.traced import pixel_theta
from test_traced import RA0, DEC0, image_wcs, linear_field


def test_beta_follows_lens_equation():
    field = linear_field(0.3, 0.1)  # beta = (1 - 0.4) theta_x, (1 - 0.2) theta_y
    ra = RA0 + 4.0 / 3600 / np.cos(np.radians(DEC0))
    dec = DEC0 - 2.5 / 3600
    q = image_fixed_quasar(ra, dec, field)
    # (ra, dec) were built with the small-angle formula, which differs from the
    # exact tangent-plane projection by ~7e-6" at 4": tolerance 1e-4".
    assert q.theta == pytest.approx((4.0, -2.5), abs=1e-4)
    assert q.beta == pytest.approx((0.6 * q.theta[0], 0.8 * q.theta[1]), abs=1e-9)
    assert "derived, fixed" in q.describe()


def test_grid_position_roundtrip():
    field = linear_field()
    w = image_wcs()
    # Put the quasar exactly at image pixel (row, col) = (7.3, 12.8).
    ra, dec = w.wcs_pix2world(12.8, 7.3, 0)
    q = image_fixed_quasar(float(ra), float(dec), field)
    assert q.grid_position(w) == pytest.approx((7.3, 12.8), abs=1e-8)
    assert q.grid_position(w, origin=(5, 10)) == pytest.approx((2.3, 2.8), abs=1e-8)
    # and theta agrees with tracing that pixel directly
    tx, ty = pixel_theta(w, field, np.array(7.3), np.array(12.8))
    assert q.theta == pytest.approx((float(tx), float(ty)), abs=1e-9)


# --- M7b: grid mapping, source-fixed and free modes --------------------------------

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from forwardgal.images import find_images  # noqa: E402
from forwardgal.quasar import free_quasar, grid_mapping, source_fixed_quasar  # noqa: E402
from test_lenstronomy import cored_isothermal_field  # noqa: E402

SHAPE = (251, 251)  # 0.1"/px, 25" wide: holds all three images of the test lens


def big_grid():
    field = cored_isothermal_field(10.0, 2.0)
    w = image_wcs(offset_arcsec=(0.0, 0.0), pixscale=0.1)
    w.wcs.crpix = [126.0, 126.0]  # grid centre on the lens centre
    return field, w, grid_mapping(w, field, origin=(0, 0), shape=SHAPE)


def small_psf():
    y, x = np.mgrid[0:15, 0:15] - 7
    k = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
    return k / k.sum()


def test_grid_mapping_matches_astropy():
    field, w, mapping = big_grid()
    rng = np.random.default_rng(6)
    for tx, ty in rng.uniform(-11, 11, (20, 2)):
        ra, dec = field.theta_to_sky(tx, ty)
        col, row = w.wcs_world2pix(ra, dec, 0)
        r, c = mapping(tx, ty)
        assert abs(float(r) - float(row)) < 1e-6 and abs(float(c) - float(col)) < 1e-6


def test_source_fixed_places_all_images():
    field, w, mapping = big_grid()
    q = source_fixed_quasar(1.3, 0.0, field, mapping, SHAPE, pad=7)
    assert len(q.images) == 3 and q.in_region == [0, 1, 2]
    unit = np.asarray(q.unit_images(small_psf(), SHAPE))
    assert unit.shape == (3, *SHAPE)
    for k, (r, c) in enumerate(q.grid_positions):
        # Fully inside the grid; 1e-5 because the toy Gaussian is truncated at its
        # stamp edge and a fractional Fourier shift spreads that step as ringing.
        assert unit[k].sum() == pytest.approx(1.0, rel=1e-5)
        rr, cc = np.unravel_index(unit[k].argmax(), SHAPE)
        assert abs(rr - r) <= 0.5 and abs(cc - c) <= 0.5  # peak at the image position
    assert "on the model grid" in q.describe()


def test_source_fixed_drops_images_off_the_grid():
    field, w, _ = big_grid()
    small = (41, 41)  # 4" around the centre: only the central image is on it
    mapping = grid_mapping(w, field, origin=(105, 105), shape=small)
    q = source_fixed_quasar(1.3, 0.0, field, mapping, small, pad=0)
    assert len(q.images) == 3 and len(q.in_region) == 1
    assert "outside the model grid" in q.describe()


def test_free_mode_follows_the_source():
    field, w, mapping = big_grid()
    q = free_quasar(1.3, 0.0, field, mapping, SHAPE, pad=7)
    # At the starting source position: the starting images, converged.
    theta, res = q.image_positions(jnp.array([1.3, 0.0]))
    np.testing.assert_allclose(np.asarray(theta), q.theta0, atol=1e-9)
    assert np.all(np.asarray(res) < 1e-8)
    # Move the source: positions equal a fresh image search there.
    beta1 = (1.36, 0.05)
    theta1, res1 = q.image_positions(jnp.array(beta1))
    fresh = np.array([im.theta for im in find_images(field, *beta1)])
    for t in np.asarray(theta1):
        assert np.min(np.hypot(*(fresh - t).T)) < 1e-6
    assert np.all(np.asarray(res1) < 1e-8)


def test_free_mode_gradient_is_inverse_jacobian():
    field, w, mapping = big_grid()
    q = free_quasar(1.3, 0.0, field, mapping, SHAPE, pad=7)
    beta = jnp.array([1.3, 0.0])
    dtheta_dbeta = jax.jacfwd(lambda b: q.image_positions(b)[0])(beta)  # (n, 2, 2)
    A = jax.vmap(jax.jacfwd(lambda t: jnp.stack(field.backtrace(t[0], t[1]))))(jnp.asarray(q.theta0))
    for k in range(q.n_images):
        np.testing.assert_allclose(np.asarray(dtheta_dbeta[k]), np.linalg.inv(np.asarray(A[k])), rtol=1e-6, atol=1e-8)


def test_free_mode_offsets_and_differentiable_images():
    field, w, mapping = big_grid()
    q = free_quasar(1.3, 0.0, field, mapping, SHAPE, pad=7)
    off = jnp.array([[0.1, -0.2], [0.0, 0.0], [0.05, 0.05]])
    t0, _ = q.image_positions(jnp.array([1.3, 0.0]))
    t1, _ = q.image_positions(jnp.array([1.3, 0.0]), off)
    np.testing.assert_allclose(np.asarray(t1 - t0), np.asarray(off), atol=1e-12)
    from forwardgal.psf import make_point_source_renderer
    render = make_point_source_renderer(small_psf(), SHAPE)
    g = jax.grad(lambda b: q.unit_images(b, None, render)[0].max())(jnp.array([1.3, 0.0]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_snap_to_psf():
    from forwardgal.quasar import snap_to_psf
    # PSF made for a source at stamp (69.829, 70.255), centre (70, 70).
    rc, d = snap_to_psf((77.823, 50.226), (69.829, 70.255), (70, 70))
    assert rc == pytest.approx((77.829, 50.255), abs=1e-9) and d == pytest.approx(0.0296, abs=1e-3)
    with pytest.raises(ValueError, match="not seem to be made"):
        snap_to_psf((77.6, 50.226), (69.829, 70.255), (70, 70))
