"""G1b: model ellipse <-> pyqtgraph EllipseROI, and the box image placement."""

import os

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402

from forwardgal.gui.geometry import (  # noqa: E402
    drag_to_ellipse, ellipse_to_roi, image_qtransform, pixel_to_theta, roi_to_ellipse)

CASES = [(-5.7, 0.2, 0.25, 0.6, 30.0), (1.0, -2.0, 0.1, 0.3, 120.0), (0.0, 0.0, 1.0, 1.0, 0.0),
         (3.0, 4.0, 0.5, 0.2, 179.0)]


@pytest.fixture(scope="module")
def app():
    return pg.mkQApp()


@pytest.mark.parametrize("ell", CASES)
def test_ellipse_to_roi_as_pyqtgraph_draws_it(app, ell):
    """The ROI pyqtgraph builds from ellipse_to_roi has the right centre and
    its major-axis tip at R_e along (sin PA, cos PA) from the centre."""
    x0, y0, R_e, q, pa = ell
    pos, size, angle = ellipse_to_roi(*ell)
    roi = pg.EllipseROI(pos=pos, size=size, angle=angle)
    c = roi.mapToParent(QPointF(size[0] / 2, size[1] / 2))
    tip = roi.mapToParent(QPointF(size[0], size[1] / 2))  # end of the u axis
    side = roi.mapToParent(QPointF(size[0] / 2, size[1]))  # end of the v axis
    assert (c.x(), c.y()) == pytest.approx((x0, y0), abs=1e-12)
    s, co = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    assert (tip.x() - x0, tip.y() - y0) == pytest.approx((R_e * s, R_e * co), abs=1e-12)
    assert np.hypot(side.x() - x0, side.y() - y0) == pytest.approx(q * R_e, abs=1e-12)


# --- Erik's inverse (fails until the TODO is written) --------------------------------


@pytest.mark.parametrize("ell", CASES)
def test_roi_to_ellipse_inverts_ellipse_to_roi(ell):
    x0, y0, R_e, q, pa = roi_to_ellipse(*ellipse_to_roi(*ell))
    assert (x0, y0, R_e, q) == pytest.approx(ell[:4], abs=1e-12)
    if ell[3] < 1:  # a circle has no PA
        assert pa == pytest.approx(ell[4] % 180, abs=1e-9)


def test_roi_taller_than_wide():
    # A ROI 1 wide, 4 tall, not rotated: major axis along +y (north): PA 0.
    x0, y0, R_e, q, pa = roi_to_ellipse((0.0, 0.0), (1.0, 4.0), 0.0)
    assert (x0, y0, R_e, q) == pytest.approx((0.5, 2.0, 2.0, 0.25))
    assert pa == pytest.approx(0.0, abs=1e-9) or pa == pytest.approx(180.0)


def test_pa_comes_back_in_0_to_180():
    for angle in (-170.0, -45.0, 100.0, 300.0):
        *_, pa = roi_to_ellipse((0.0, 0.0), (2.0, 1.0), angle)
        assert 0.0 <= pa < 180.0
        assert pa == pytest.approx((90.0 - angle) % 180.0)


# --- image placement -----------------------------------------------------------------


def test_pixel_to_theta_places_pixel_centres(app):
    from types import SimpleNamespace

    # a rotated, flipped pixel grid: theta = A (col, row) + t exactly
    A = np.array([[-0.03 * np.cos(0.3), 0.03 * np.sin(0.3)], [0.03 * np.sin(0.3), 0.03 * np.cos(0.3)]])
    t = np.array([12.0, -4.0])
    rows, cols = np.indices((7, 9))
    k = np.array([-0.2, 0.0, 0.2])  # symmetric subpixel offsets
    sub_c = cols[..., None] + np.repeat(k, 3)[None, None]
    sub_r = rows[..., None] + np.tile(k, 3)[None, None]
    grid = SimpleNamespace(theta_sub_x=A[0, 0] * sub_c + A[0, 1] * sub_r + t[0],
                           theta_sub_y=A[1, 0] * sub_c + A[1, 1] * sub_r + t[1])
    A2, t2, rms = pixel_to_theta(grid)
    np.testing.assert_allclose(A2, A, atol=1e-12)
    np.testing.assert_allclose(t2, t, atol=1e-12)
    assert rms < 1e-12
    tr = image_qtransform(A2, t2)
    p = tr.map(QPointF(4 + 0.5, 2 + 0.5))  # centre of pixel [row 2, col 4]
    assert (p.x(), p.y()) == pytest.approx(tuple(A @ [4, 2] + t), abs=1e-12)


@pytest.mark.parametrize("p1, pa", [
    ((0.0, 1.0), 0.0),     # drag north: major axis north-south, PA 0
    ((1.0, 0.0), 90.0),    # drag east: PA 90
    ((1.0, 1.0), 45.0),    # north-east
    ((-1.0, 1.0), 135.0),  # north-west: PA -45 = 135
    ((0.0, -1.0), 0.0),    # drag south: the same axis as north
])
def test_drag_to_ellipse_pa_east_of_north(p1, pa):
    x0, y0, R_e, q, got = drag_to_ellipse((0.0, 0.0), p1)
    assert got == pytest.approx(pa) and q == 0.6
    assert (x0, y0) == pytest.approx((p1[0] / 2, p1[1] / 2)) and R_e == pytest.approx(np.hypot(*p1) / 2)


def test_drag_tips_are_the_drawn_ellipses_major_axis_ends():
    """The ellipse drawn for the result (ellipse_to_roi) has its major-axis tips where the drag began and ended."""
    p0, p1 = (2.0, -1.0), (2.6, -0.2)
    x0, y0, R_e, q, pa = drag_to_ellipse(p0, p1, q=0.5)
    major = np.array([np.sin(np.radians(pa)), np.cos(np.radians(pa))])
    tips = {tuple(np.round(np.array([x0, y0]) + s * R_e * major, 9)) for s in (-1, 1)}
    assert tips == {tuple(np.round(p0, 9)), tuple(np.round(p1, 9))}


@pytest.mark.parametrize("m", [2, 3, 4, 5])
@pytest.mark.parametrize("pa, pa_ref", [(30.0, 30.0), (30.0, 200.0), (150.0, 10.0)])
def test_bend_handle_sits_on_the_ridge_of_that_mode(m, pa, pa_ref):
    """The handle is where mode m alone puts the ridge (v' = 0 in the profile code), and dragging round-trips."""
    import jax.numpy as jnp

    from forwardgal.gui.geometry import HANDLE_U, bend_from_handle, bend_handle_pos
    from forwardgal.profiles.sersic import Bending, bend_minor

    x0, y0, R_e, b = -5.7, 0.2, 0.3, 0.04
    x, y = bend_handle_pos(x0, y0, R_e, pa, pa_ref, m, b)
    p = np.radians(pa)
    u = (x - x0) * np.sin(p) + (y - y0) * np.cos(p)
    v = (x - x0) * np.cos(p) - (y - y0) * np.sin(p)
    assert u == pytest.approx(R_e * HANDLE_U[m])
    assert float(bend_minor(jnp.asarray(u), jnp.asarray(v), R_e, pa, Bending((m,), [b], pa_ref))) == pytest.approx(
        0.0, abs=1e-12)
    assert bend_from_handle(x0, y0, R_e, pa, pa_ref, m, x, y) == pytest.approx(b)
    # dragging along the major axis does not change b (only the part along v counts)
    assert bend_from_handle(x0, y0, R_e, pa, pa_ref, m, x + 0.1 * np.sin(p), y + 0.1 * np.cos(p)) == pytest.approx(b)
    assert bend_handle_pos(x0, y0, R_e, pa, pa_ref, m, 0.0) == pytest.approx(
        (x0 + R_e * HANDLE_U[m] * np.sin(p), y0 + R_e * HANDLE_U[m] * np.cos(p)))  # b = 0: on the major axis
