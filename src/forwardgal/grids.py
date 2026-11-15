"""Pixel grids, subsampling and binning.

Conventions: 0-based [row, col] = [y, x]. A data pixel with integer index j
covers [j - 0.5, j + 0.5) in pixel units; its centre is at j.
"""

from __future__ import annotations

import numpy as np


def subpixel_centres(ny, nx, s):
    """Fractional [row, col] of the centres of s x s subpixels per data pixel.

    Parameters
    ----------
    ny, nx : int
        Shape of the data-pixel grid.
    s : int
        Subsampling factor (s = 1: data-pixel centres).

    Returns
    -------
    rows, cols : (s*ny, s*nx) arrays
        Subpixel k (0 <= k < s) of pixel j sits at j + (k + 0.5)/s - 0.5.
    """
    r = (np.arange(s * ny) + 0.5) / s - 0.5
    c = (np.arange(s * nx) + 0.5) / s - 0.5
    rows, cols = np.meshgrid(r, c, indexing="ij")
    return rows, cols


def bin_mean(img, s):
    """Average s x s blocks of a subsampled image back to data pixels.

    Parameters
    ----------
    img : (s*ny, s*nx) array (numpy or JAX)
    s : int

    Returns
    -------
    (ny, nx) array: the mean of each block. Use the mean (not sum) for
    surface-brightness units such as MJy/sr.
    """
    ny, nx = img.shape[0] // s, img.shape[1] // s
    return img.reshape(ny, s, nx, s).mean(axis=(1, 3))


def local_theta(rows, cols, pixscale, row0, col0):
    """Pixel [row, col] -> local (theta_x, theta_y) in arcsec for a north-up image.

    theta_x = -pixscale * (col - col0)   (+east = towards smaller col)
    theta_y = +pixscale * (row - row0)   (+north = towards larger row)

    Only valid for an unrotated, north-up, east-left grid; used for tests and
    unlensed rendering. Real data use the WCS (see forwardgal.lens).
    """
    return -pixscale * (np.asarray(cols) - col0), pixscale * (np.asarray(rows) - row0)
