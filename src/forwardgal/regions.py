"""Which pixels are fit and which are rendered.

Three nested sets of pixels, all in 0-based [row, col] of the cutout:

- fit pixels    : the user's mask. Only these enter chi^2.
- render pixels : every pixel within ``margin`` pixels (Euclidean distance)
                  of a fit pixel. The model is computed here, because the PSF
                  spreads light from these pixels into the fit pixels.
- render box    : the smallest rectangle containing the render pixels. PSF
                  convolution (FFT) needs a rectangle; box pixels that are not
                  render pixels are set to zero.

This mirrors GALFIT, which computes and convolves its model over the whole
fitting rectangle (including under bad-pixel-masked pixels) but evaluates
chi^2 only on unmasked pixels. No data outside the mask are ever used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class FitRegion:
    """Fit pixels, render pixels and render box (see module docstring).

    Attributes
    ----------
    mask : (ny, nx) bool array
        Fit pixels, on the full cutout.
    render : (ny, nx) bool array
        Render pixels, on the full cutout (includes ``mask``).
    margin : float
        Requested margin in pixels.
    box : (r0, r1, c0, c1)
        Render box as slices [r0:r1, c0:c1] of the cutout.
    margin_available : dict
        For each side ("north", "south", "east", "west"), the distance in
        pixels from the outermost fit pixel to the cutout edge on that side,
        capped at ``margin``. Less than ``margin`` means the cutout edge
        truncates the margin there.
    """

    mask: np.ndarray
    render: np.ndarray
    margin: float
    box: tuple
    margin_available: dict

    @property
    def box_shape(self):
        r0, r1, c0, c1 = self.box
        return (r1 - r0, c1 - c0)

    def in_box(self, image):
        """Cut a full-cutout image down to the render box."""
        r0, r1, c0, c1 = self.box
        return image[r0:r1, c0:c1]

    def render_pixels_in_box(self):
        """(rows, cols) of render pixels, 0-based relative to the box corner."""
        rows, cols = np.nonzero(self.in_box(self.render))
        return rows, cols

    def describe(self):
        """Human-readable summary (also saved with each fit)."""
        n_fit, n_render = int(self.mask.sum()), int(self.render.sum())
        bh, bw = self.box_shape
        lines = [
            f"fit pixels (mask)    : {n_fit}",
            f"render pixels        : {n_render}  (mask + {self.margin:g} px margin)",
            f"render box           : rows {self.box[0]}:{self.box[1]}, cols {self.box[2]}:{self.box[3]}"
            f"  ({bh} x {bw} = {bh * bw} px)",
            f"cutout               : {self.mask.shape[0]} x {self.mask.shape[1]} = {self.mask.size} px",
        ]
        short = {k: v for k, v in self.margin_available.items() if v < self.margin}
        if short:
            sides = ", ".join(f"{k} {v:g} px" for k, v in short.items())
            lines.append(f"WARNING margin truncated by the cutout edge: {sides} (requested {self.margin:g} px)")
        else:
            lines.append("margin complete on all sides")
        return "\n".join(lines)


def polygon_mask(shape, vertices):
    """Boolean mask of the pixels whose centres lie inside a polygon.

    Parameters
    ----------
    shape : (ny, nx)
    vertices : (M, 2) array
        Polygon corners as 0-based (row, col) pixel coordinates (the GUI
        will provide these from the drawn outline).

    Returns
    -------
    (ny, nx) bool array.
    """
    ny, nx = shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    path = Path(np.asarray(vertices, float)[:, ::-1])  # Path wants (x=col, y=row)
    inside = path.contains_points(np.column_stack([cols.ravel(), rows.ravel()]))
    return inside.reshape(shape)


def fit_region(mask, margin=35.0):
    """Build the FitRegion for a mask.

    Parameters
    ----------
    mask : (ny, nx) bool array
        Fit pixels.
    margin : float
        Render pixels are those within this distance (pixels) of a fit pixel.
        Default 35 px (~1" at 0.03"/px): with the F115W PSF, < 0.5% of a fit
        pixel's convolved light then comes from beyond the margin.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("mask is empty")
    # distance_transform_edt gives, for every pixel, the distance to the
    # nearest zero; on ~mask that is the distance to the nearest fit pixel.
    render = distance_transform_edt(~mask) <= margin

    rows, cols = np.nonzero(render)
    box = (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)

    ny, nx = mask.shape
    mr, mc = np.nonzero(mask)
    # North = larger row, east = smaller col (north-up, east-left cutout).
    available = {
        "south": float(mr.min()),
        "north": float(ny - 1 - mr.max()),
        "east": float(mc.min()),
        "west": float(nx - 1 - mc.max()),
    }
    available = {k: min(v, float(margin)) for k, v in available.items()}
    return FitRegion(mask=mask, render=render, margin=float(margin), box=box, margin_available=available)


def ellipse_polygon(centre_rc, a, b, pa_deg, n=72):
    """Vertices (row, col) of an ellipse, for ``polygon_mask``.

    Parameters
    ----------
    centre_rc : (row, col)
        Centre in 0-based pixel coordinates.
    a, b : float
        Semi-major and semi-minor axes in pixels.
    pa_deg : float
        Position angle of the major axis, east of north, for a north-up,
        east-left image: north = +row, east = -col.
    n : int
        Number of vertices.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pa = np.radians(pa_deg)
    major = np.array([np.cos(pa), -np.sin(pa)])  # (row, col) direction of the major axis
    minor = np.array([np.sin(pa), np.cos(pa)])
    pts = np.outer(a * np.cos(t), major) + np.outer(b * np.sin(t), minor)
    return pts + np.asarray(centre_rc, float)
