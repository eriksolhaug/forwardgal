import jax
import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.geometry import polygon_signed_area, ray_polygon_range

UNIT_SQUARE = jnp.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def test_signed_area():
    assert float(polygon_signed_area(UNIT_SQUARE)) == pytest.approx(1.0)
    assert float(polygon_signed_area(UNIT_SQUARE[::-1])) == pytest.approx(-1.0)


def test_origin_inside_square():
    phi = jnp.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    t_in, t_out = ray_polygon_range(UNIT_SQUARE, phi)
    np.testing.assert_allclose(t_in, 0.0)
    np.testing.assert_allclose(t_out, [0.5, 0.5 * np.sqrt(2), 0.5, 0.5], rtol=1e-12)


def test_origin_outside_square():
    square = jnp.array([[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5]])
    t_in, t_out = ray_polygon_range(square, jnp.array([0.0, np.pi / 2, np.pi]))
    np.testing.assert_allclose(t_in, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(t_out, [2.0, 0.0, 0.0], atol=1e-12)  # misses -> 0


def test_winding_order_does_not_matter():
    phi = jnp.linspace(0, 2 * np.pi, 37)
    quad = jnp.array([[0.3, 0.1], [1.4, 0.4], [1.1, 1.3], [0.2, 0.9]])
    a = ray_polygon_range(quad, phi)
    b = ray_polygon_range(quad[::-1], phi)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_ray_along_edge_gives_no_nan():
    square = jnp.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    t_in, t_out = ray_polygon_range(square, jnp.array([0.0]))  # along y = 0 edge
    assert np.all(np.isfinite(t_in)) and np.all(np.isfinite(t_out))


def random_convex_quad(rng, contains_origin):
    """A random convex quadrilateral: 4 points on a jittered ellipse, in angle order."""
    ang = np.sort(rng.uniform(0, 2 * np.pi, 4))
    r = rng.uniform(0.5, 1.0)
    pts = np.stack([r * np.cos(ang), 0.6 * r * np.sin(ang)], axis=1)
    shift = rng.uniform(-0.2, 0.2, 2) if contains_origin else np.array([2.5, -1.0])
    return jnp.array(pts + shift)


@pytest.mark.parametrize("contains_origin", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_polar_area_equals_shoelace(seed, contains_origin):
    # Area = 1/2 * integral over phi of (t_out^2 - t_in^2): independent check of
    # the geometry against the shoelace formula.
    quad = random_convex_quad(np.random.default_rng(seed), contains_origin)
    nphi = 20000
    phi = (jnp.arange(nphi) + 0.5) / nphi * 2 * np.pi
    t_in, t_out = ray_polygon_range(quad, phi)
    area = 0.5 * float(jnp.sum(t_out**2 - t_in**2)) * (2 * np.pi / nphi)
    assert area == pytest.approx(abs(float(polygon_signed_area(quad))), rel=1e-4)


def test_gradients_are_finite():
    phi = jnp.linspace(0, 2 * np.pi, 64, endpoint=False)
    square = jnp.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    g = jax.grad(lambda v: jnp.sum(ray_polygon_range(v, phi)[1]))(square)
    assert np.all(np.isfinite(g))
