"""Display geometry for the GUI panels (G1b): no physics here.

Ellipses
--------
A model ellipse is (x0, y0, R_e, q, pa_deg): centre in arcsec (+x east,
+y north), semi-major axis R_e (arcsec), axis ratio q = b/a <= 1, PA in
degrees east of north (the major axis points along (sin PA, cos PA)).

A pyqtgraph EllipseROI is (pos, size, angle): the ellipse inscribed in the
rectangle 0 <= u <= w, 0 <= v <= h of the ROI's own frame, placed in the
plot's data coordinates (x, y) by

    (x, y) = pos + R(angle) (u, v),   R(a) = [[cos a, -sin a], [sin a, cos a]],

i.e. ``angle`` (degrees) turns counter-clockwise from +x towards +y in data
coordinates (checked with roi.mapToParent; an inverted x axis on screen
does not change it). The ellipse centre is pos + R(angle) (w/2, h/2).

Image placement
---------------
``pixel_to_theta`` fits the affine map from box pixel [row, col] (0-based
centres) to theta (arcsec) and returns it as a QTransform for a pyqtgraph
ImageItem in row-major order (item coordinates: x = col, y = row, pixel
[r, c] covering c..c+1, r..r+1).
"""

from __future__ import annotations

import numpy as np


def _rot(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def ellipse_to_roi(x0, y0, R_e, q, pa_deg):
    """Model ellipse -> EllipseROI (pos (x, y), size (w, h), angle_deg).

    The major axis lies along the ROI's u axis: w = 2 R_e, h = 2 q R_e. Its
    direction (sin PA, cos PA) has angle 90 - PA from +x.
    """
    w, h = 2.0 * R_e, 2.0 * q * R_e
    angle = 90.0 - pa_deg
    pos = np.array([x0, y0]) - _rot(angle) @ np.array([w / 2, h / 2])
    return (float(pos[0]), float(pos[1])), (float(w), float(h)), float(angle)


def roi_to_ellipse(pos, size, angle_deg):
    """EllipseROI (pos, size, angle_deg) -> model ellipse (x0, y0, R_e, q, pa_deg).

    The inverse of ``ellipse_to_roi``, for any ROI the user can make by
    dragging: also when h > w (then the major axis lies along the ROI's v
    axis, at angle_deg + 90 from +x). Return pa_deg in [0, 180) and q <= 1.
    """
    # TODO(Erik): write the inverse.
    #   1. The centre: pos + R(angle) (w/2, h/2). Use _rot (above); pos and
    #      size come in as pairs (x, y) and (w, h).
    #   2. R_e and q from (w, h): the semi-major axis is half the LARGER of w
    #      and h; q is smaller / larger.
    #   3. The major axis direction, as an angle from +x: angle_deg if w >= h,
    #      angle_deg + 90 if h > w. PA = 90 - (that angle) (see the module
    #      docstring), then bring it into [0, 180) with % 180.
    # Check with:  pytest tests/test_gui_geometry.py -q
    w, h = size
    w, h = abs(w), max(abs(h), 1e-12) if w == 0 else abs(h)
    x0, y0 = np.asarray(pos) + _rot(angle_deg) @ np.array([w / 2, h / 2])
    R_e = max(w, h) / 2
    q = min(w, h) / max(w, h)
    major = angle_deg if w>= h else angle_deg + 90.0
    pa_deg = (90.0 - major) % 180.0
    return (float(x0), float(y0), float(R_e), float(q), float(pa_deg))


def drag_to_ellipse(p0, p1, q=0.6):
    """A mouse drag from one tip of the major axis to the other -> model ellipse (x0, y0, R_e, q, pa_deg).

    p0, p1 : (x, y) in arcsec (data coordinates: +x east, +y north).
    The centre is the midpoint, R_e half the drag length, and since the major
    axis points along (sin PA, cos PA), PA = atan2(dx, dy) brought into
    [0, 180). q is not set by the drag (the ROI's handle changes it after).
    """
    (xa, ya), (xb, yb) = p0, p1
    dx, dy = xb - xa, yb - ya
    R_e = float(np.hypot(dx, dy) / 2)
    pa = float(np.degrees(np.arctan2(dx, dy)) % 180.0)
    return float((xa + xb) / 2), float((ya + yb) / 2), R_e, float(q), pa


def pixel_to_theta(grid):
    """Affine map box pixel -> theta from a TracedGrid.

    Returns (A, t, rms): theta = A @ (col, row) + t for 0-based pixel
    centres (A is 2 x 2, arcsec per pixel; t in arcsec), fitted by least
    squares to the grid's traced pixel centres, and the rms misfit (arcsec;
    the tangent-plane distortion over a box is tiny).
    """
    tx = np.asarray(grid.theta_sub_x).mean(axis=-1)  # pixel centres (subpixels are symmetric)
    ty = np.asarray(grid.theta_sub_y).mean(axis=-1)
    rows, cols = np.indices(tx.shape)
    X = np.column_stack([cols.ravel(), rows.ravel(), np.ones(tx.size)])
    cx, *_ = np.linalg.lstsq(X, tx.ravel(), rcond=None)
    cy, *_ = np.linalg.lstsq(X, ty.ravel(), rcond=None)
    A = np.array([cx[:2], cy[:2]])
    t = np.array([cx[2], cy[2]])
    rms = float(np.sqrt(np.mean((X @ cx - tx.ravel()) ** 2 + (X @ cy - ty.ravel()) ** 2)))
    return A, t, rms


def image_qtransform(A, t):
    """QTransform placing a row-major ImageItem of the box in theta coordinates.

    Item point (x, y) = (col + 0.5, row + 0.5) is pixel [row, col]'s centre,
    which must land on A @ (col, row) + t; Qt maps x' = m11 x + m21 y + dx,
    y' = m12 x + m22 y + dy.
    """
    from PySide6.QtGui import QTransform

    d = t - A @ np.array([0.5, 0.5])
    return QTransform(A[0, 0], A[1, 0], A[0, 1], A[1, 1], d[0], d[1])


# --- bending handles (triangles; G6a, 2026-11-23) --------------------------------------------
#
# A bent Sersic's ridge (where the bent minor coordinate v' = 0; profiles.sersic.bend_minor) lies at
#     v_ridge(u) = s sum_m b_m (-s u / R_e)^m,   s = sign cos(PA - pa_ref),
# with u along the major axis (sin PA, cos PA) and v along (cos PA, -sin PA) (arcsec, x east, y north).
# Mode m's handle sits at u_m = R_e * HANDLE_U[m] (beyond the major-axis tip; staggered so that the
# handles do not overlap at b = 0), at the ridge offset mode m alone gives there; it moves only along v.

HANDLE_U = {2: 1.0, 3: 1.25, 4: 1.5, 5: 1.75}


def _bend_factor(m, pa_deg, pa_ref):
    """v of mode m's handle per unit b_m: s (-s u_m / R_e)^m."""
    s = 1.0 if np.cos(np.radians(pa_deg - pa_ref)) >= 0 else -1.0
    return s * (-s * HANDLE_U[m]) ** m


def bend_handle_pos(x0, y0, R_e, pa_deg, pa_ref, m, b):
    """(x, y) in arcsec of the handle of bending mode m with amplitude b (arcsec)."""
    pa = np.radians(pa_deg)
    u, v = R_e * HANDLE_U[m], b * _bend_factor(m, pa_deg, pa_ref)
    return (float(x0 + u * np.sin(pa) + v * np.cos(pa)), float(y0 + u * np.cos(pa) - v * np.sin(pa)))


def bend_from_handle(x0, y0, R_e, pa_deg, pa_ref, m, x, y):
    """b_m (arcsec) for a handle dragged to (x, y): its v (the part along the minor axis) / the factor."""
    pa = np.radians(pa_deg)
    v = (x - x0) * np.cos(pa) - (y - y0) * np.sin(pa)
    return float(v / _bend_factor(m, pa_deg, pa_ref))
