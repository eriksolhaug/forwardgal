"""Critical curves and caustics from deflection maps (numpy; not differentiated).

The lens mapping beta = theta - alpha(theta) has Jacobian

    A = d beta / d theta = [[1 - dax/dx, -dax/dy],
                            [-day/dx,    1 - day/dy]],

with x = theta_x (east), y = theta_y (north). The magnification is
mu = 1 / det A. Critical curves are where det A = 0 (image plane); caustics
are their images beta = theta - alpha(theta) (source plane).

Derivatives are central differences on the deflection-map grid. One column
step changes theta_x by ``step_x`` (negative for a north-up, east-left map),
so d/dtheta_x = (d/dcol) / step_x; likewise d/dtheta_y = (d/drow) / step_y.
Accuracy is therefore set by the map's pixel scale (0.126" for the COOL J1153
maps); contours are linearly interpolated between grid points.
"""

from __future__ import annotations

import numpy as np
from contourpy import contour_generator


def jacobian_det(field):
    """det A on the deflection-map grid, (My, Mx) array.

    Uses numpy.gradient: second-order central differences inside the map,
    first-order one-sided differences on its edges.
    """
    dax_drow, dax_dcol = np.gradient(field.alpha_x)
    day_drow, day_dcol = np.gradient(field.alpha_y)
    dax_dx, dax_dy = dax_dcol / field.step_x, dax_drow / field.step_y
    day_dx, day_dy = day_dcol / field.step_x, day_drow / field.step_y
    return (1.0 - dax_dx) * (1.0 - day_dy) - dax_dy * day_dx


def critical_curves(field, det=None):
    """Critical curves as a list of (N, 2) arrays of (theta_x, theta_y) in arcsec.

    Each array is one connected curve (closed curves repeat their first point).
    ``det`` may be passed to reuse a precomputed ``jacobian_det(field)``.
    """
    if det is None:
        det = jacobian_det(field)
    # contourpy works in array-index coordinates: x = column, y = row.
    lines = contour_generator(z=det).lines(0.0)
    curves = []
    for line in lines:
        cols, rows = line[:, 0], line[:, 1]
        tx, ty = field.pixel_to_theta(rows, cols)
        curves.append(np.column_stack([tx, ty]))
    return curves


def caustics(field, curves):
    """Map critical curves to the source plane: list of (N, 2) (beta_x, beta_y) arrays."""
    out = []
    for c in curves:
        bx, by = field.backtrace(c[:, 0], c[:, 1])
        out.append(np.column_stack([bx, by]))
    return out
