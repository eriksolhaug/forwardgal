"""Image pixels traced to the source plane (precomputed once per cutout).

The lens model is fixed and the data pixels do not move, so the source-plane
positions beta = theta - alpha(theta) of all subpixel centres and all pixel
corners are computed once here (numpy) and passed to the JAX renderer as
constant arrays.

Coordinates: data pixel [row, col] (0-based, fractional) -> RA, Dec via the
image WCS -> theta (arcsec) via the deflection maps' tangent plane
(``DeflectionField.sky_to_theta``) -> beta via ``DeflectionField.backtrace``.
theta_x, beta_x point east; theta_y, beta_y point north.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forwardgal.geometry import polygon_signed_area


@dataclass(frozen=True)
class TracedGrid:
    """Source-plane positions of an image grid's subpixels and pixel corners.

    Attributes
    ----------
    shape : (ny, nx)
        Data-pixel shape of the traced region.
    subsample : int
        s; there are s x s subpixels per data pixel.
    beta_sub_x, beta_sub_y : (ny, nx, s*s) arrays
        beta (arcsec) of each pixel's subpixel centres.
    theta_sub_x, theta_sub_y : (ny, nx, s*s) arrays
        theta (arcsec) of the same subpixel centres (not traced): used for
        image-plane (unlensed) components.
    theta_corner_x, theta_corner_y : (ny+1, nx+1) arrays
        theta (arcsec) of the pixel corners; corner [i, j] is at fractional
        pixel position (row, col) = (i - 0.5, j - 0.5), shared by up to four
        pixels.
    beta_corner_x, beta_corner_y : (ny+1, nx+1) arrays
        beta (arcsec) of the pixel corners. Pixel [r, c] maps to the
        quadrilateral through corners [r, c], [r, c+1], [r+1, c+1], [r+1, c].
    """

    shape: tuple
    subsample: int
    beta_sub_x: np.ndarray
    beta_sub_y: np.ndarray
    theta_sub_x: np.ndarray
    theta_sub_y: np.ndarray
    theta_corner_x: np.ndarray
    theta_corner_y: np.ndarray
    beta_corner_x: np.ndarray
    beta_corner_y: np.ndarray

    def coordinates(self, plane):
        """(sub_x, sub_y, corner_x, corner_y) in the "source" (beta) or "image" (theta) plane."""
        if plane == "source":
            return self.beta_sub_x, self.beta_sub_y, self.beta_corner_x, self.beta_corner_y
        if plane == "image":
            return self.theta_sub_x, self.theta_sub_y, self.theta_corner_x, self.theta_corner_y
        raise ValueError(f"plane must be 'source' or 'image', got {plane!r}")

    def footprint(self, r, c):
        """(4, 2) source-plane corners (beta_x, beta_y) of pixel [r, c]."""
        rr = [r, r, r + 1, r + 1]
        cc = [c, c + 1, c + 1, c]
        return np.column_stack([self.beta_corner_x[rr, cc], self.beta_corner_y[rr, cc]])

    def area_ratio(self):
        """Signed (footprint area) / (pixel area) for every pixel, (ny, nx).

        For a pixel small compared with changes in the lens mapping this is
        det(d beta / d theta) = 1 / mu (signed magnification). Negative values
        mean the footprint is mirror-imaged (parity flip).
        """
        ny, nx = self.shape
        out = np.empty((ny, nx))
        for r in range(ny):
            for c in range(nx):
                rr = [r, r, r + 1, r + 1]
                cc = [c, c + 1, c + 1, c]
                img = np.column_stack([self.theta_corner_x[rr, cc], self.theta_corner_y[rr, cc]])
                out[r, c] = float(polygon_signed_area(self.footprint(r, c))) / float(polygon_signed_area(img))
        return out


def pixel_theta(image_wcs, field, rows, cols):
    """theta (arcsec) of fractional 0-based image pixel positions (any shape)."""
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    ra, dec = image_wcs.wcs_pix2world(cols.ravel(), rows.ravel(), 0)
    tx, ty = field.sky_to_theta(ra, dec)
    return np.asarray(tx).reshape(rows.shape), np.asarray(ty).reshape(rows.shape)


def trace_grid(image_wcs, shape, field, subsample=5, origin=(0, 0)):
    """Trace an image region's subpixels and pixel corners to the source plane.

    Parameters
    ----------
    image_wcs : astropy.wcs.WCS
        Celestial WCS of the image (e.g. ``ImageData.wcs`` of the cutout).
    shape : (ny, nx)
        Number of data pixels to trace.
    field : DeflectionField
        Already rescaled to the source redshift if needed.
    subsample : int
        s x s subpixels per pixel (must match RenderSettings.subsample).
    origin : (r0, c0)
        Image pixel of the traced region's [0, 0] (e.g. a FitRegion box corner).

    Returns
    -------
    TracedGrid
    """
    ny, nx = shape
    r0, c0 = origin
    s = subsample

    k = (np.arange(s) + 0.5) / s - 0.5  # subpixel offsets
    dr, dc = np.meshgrid(k, k, indexing="ij")
    pr, pc = np.mgrid[0:ny, 0:nx]
    sub_r = (pr[..., None] + r0) + dr.ravel()  # (ny, nx, s*s)
    sub_c = (pc[..., None] + c0) + dc.ravel()
    tx, ty = pixel_theta(image_wcs, field, sub_r, sub_c)
    bsx, bsy = field.backtrace(tx, ty)

    cr, cc = np.mgrid[0 : ny + 1, 0 : nx + 1]
    tcx, tcy = pixel_theta(image_wcs, field, cr + r0 - 0.5, cc + c0 - 0.5)
    bcx, bcy = field.backtrace(tcx, tcy)

    if not (np.all(np.isfinite(bsx)) and np.all(np.isfinite(bcx))):
        raise ValueError("part of the image region lies outside the deflection maps")

    return TracedGrid(
        shape=(ny, nx),
        subsample=s,
        beta_sub_x=np.asarray(bsx),
        beta_sub_y=np.asarray(bsy),
        theta_sub_x=np.asarray(tx),
        theta_sub_y=np.asarray(ty),
        theta_corner_x=np.asarray(tcx),
        theta_corner_y=np.asarray(tcy),
        beta_corner_x=np.asarray(bcx),
        beta_corner_y=np.asarray(bcy),
    )
