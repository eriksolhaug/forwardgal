"""Fixed lens model given as deflection maps (e.g. Lenstool ``dpl`` output).

Coordinates
-----------
theta (image plane) and beta (source plane) are positions in arcsec in the
tangent plane of the deflection maps' TAN projection, whose reference point
(CRVAL1, CRVAL2) is the lens-model reference position:

    theta_x : +east  (+RA,  to the left in a north-up image)
    theta_y : +north (+Dec, up)

On the deflection-map grid, with 0-based array indices [row, col] = [i, j],

    theta_x = 3600 * CDELT1 * (j - j0),   theta_y = 3600 * CDELT2 * (i - i0),
    (i0, j0) = (CRPIX2 - 1, CRPIX1 - 1),

which is exact for a TAN projection evaluated at its own reference point.

Lenstool's dplx: Lenstool's internal x-axis increases to the WEST, so its dplx
map is the deflection along -theta_x. ``load_deflection`` therefore stores
alpha_x = -dplx by default (``x_points_west=True``). This was verified on the
COOL J1153 model: with the flip, quasar images 1A, 1B, 1C back-trace to one
source position (rms 0.1"); without it they scatter by 18".

Sky frames: Lenstool labels its WCS FK5 but never transforms between frames, so
RA/Dec from any input (e.g. ICRS JWST mosaics) are used as given, without an
FK5 <-> ICRS conversion (which would shift positions by up to ~0.02").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from forwardgal.cosmology import deflection_scale


@dataclass
class DeflectionField:
    """Deflection maps alpha_x, alpha_y on a regular theta grid.

    Attributes
    ----------
    alpha_x, alpha_y : (My, Mx) float64 arrays
        Deflection in arcsec along +theta_x (east) and +theta_y (north),
        indexed [row, col]. Lens equation: beta = theta - alpha(theta).
    wcs : astropy.wcs.WCS
        TAN WCS of the maps; CRVAL is the reference position (theta = 0).
    step_x, step_y : float
        Change in theta_x / theta_y per +1 column / +1 row, in arcsec.
        step_x < 0 for a north-up, east-left map.
    i0, j0 : float
        0-based [row, col] of the reference pixel (where theta = 0).
    z_ref : float
        Source redshift for which the maps were computed.
    """

    alpha_x: np.ndarray
    alpha_y: np.ndarray
    wcs: WCS
    step_x: float
    step_y: float
    i0: float
    j0: float
    z_ref: float

    def pixel_to_theta(self, row, col):
        """0-based map pixel [row, col] -> (theta_x, theta_y) in arcsec.

        Accepts scalars or arrays (non-integer pixel positions are fine).
        """
        theta_x = self.step_x * (np.asarray(col) - self.j0)
        theta_y = self.step_y * (np.asarray(row) - self.i0)
        return theta_x, theta_y

    def theta_to_pixel(self, theta_x, theta_y):
        """(theta_x, theta_y) in arcsec -> 0-based map pixel (row, col), as floats."""
        col = np.asarray(theta_x) / self.step_x + self.j0
        row = np.asarray(theta_y) / self.step_y + self.i0
        return row, col

    def alpha(self, theta_x, theta_y):
        """Deflection (alpha_x, alpha_y) in arcsec at arbitrary theta (arcsec).

        Bilinear interpolation between map pixels. Accepts scalars or arrays of
        any (matching) shape. Positions outside the map give NaN.
        """
        row, col = self.theta_to_pixel(theta_x, theta_y)
        return bilinear(self.alpha_x, row, col), bilinear(self.alpha_y, row, col)

    def backtrace(self, theta_x, theta_y):
        """Lens equation: source-plane beta = theta - alpha(theta), in arcsec."""
        alpha_x, alpha_y = self.alpha(theta_x, theta_y)
        return np.asarray(theta_x) - alpha_x, np.asarray(theta_y) - alpha_y

    def rescaled(self, z_lens, z_src, cosmo):
        """Copy of this field with alpha converted from z_ref to z_src.

        Multiplies both maps by f = (D_ls/D_s)(z_src) / (D_ls/D_s)(z_ref);
        see forwardgal.cosmology. The returned field has z_ref = z_src.
        """
        f = deflection_scale(z_lens, self.z_ref, z_src, cosmo)
        return replace(
            self, alpha_x=f * self.alpha_x, alpha_y=f * self.alpha_y, z_ref=float(z_src)
        )

    def sky_to_theta(self, ra_deg, dec_deg):
        """RA, Dec (deg) -> (theta_x, theta_y) in arcsec, via the maps' TAN plane."""
        # wcs_world2pix with origin=0 returns 0-based (col, row) = (x, y).
        col, row = self.wcs.wcs_world2pix(
            np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float), 0
        )
        return self.pixel_to_theta(row, col)

    def theta_to_sky(self, theta_x, theta_y):
        """(theta_x, theta_y) in arcsec -> RA, Dec in deg."""
        row, col = self.theta_to_pixel(theta_x, theta_y)
        ra, dec = self.wcs.wcs_pix2world(col, row, 0)
        return ra, dec


def bilinear(arr, row, col):
    """Bilinear interpolation of a 2-D array at fractional 0-based [row, col].

    Parameters
    ----------
    arr : (Ny, Nx) array
    row, col : scalars or arrays of the same shape
        Fractional pixel positions; integer values hit pixel centres exactly.

    Returns
    -------
    array with the shape of ``row``; NaN where the position lies outside
    [0, Ny-1] x [0, Nx-1].
    """
    row = np.asarray(row, dtype=float)
    col = np.asarray(col, dtype=float)
    ny, nx = arr.shape
    inside = (row >= 0) & (row <= ny - 1) & (col >= 0) & (col <= nx - 1)

    # Lower-left grid point. Clipping to ny-2 / nx-2 lets a position exactly on
    # the last row/column use weight f = 1 on that edge instead of indexing past it.
    # Outside points are clipped too (so indexing is safe) and masked at the end.
    i = np.clip(np.floor(row).astype(int), 0, ny - 2)
    j = np.clip(np.floor(col).astype(int), 0, nx - 2)
    fr = row - i
    fc = col - j

    value = (
        (1 - fr) * (1 - fc) * arr[i, j]
        + (1 - fr) * fc * arr[i, j + 1]
        + fr * (1 - fc) * arr[i + 1, j]
        + fr * fc * arr[i + 1, j + 1]
    )
    return np.where(inside, value, np.nan)


def _read_map(path):
    with fits.open(path) as hdul:
        hdu = hdul[0]
        # astype(float64) copies into native byte order, so the file can close.
        return hdu.data.astype(np.float64), hdu.header


def load_deflection(path_x, path_y, z_ref, x_points_west=True):
    """Load Lenstool-style deflection maps (arcsec) with their WCS.

    Parameters
    ----------
    path_x, path_y : str or Path
        FITS files with the deflection along x and y (Lenstool dplx, dply),
        in arcsec, image in the primary HDU.
    z_ref : float
        Source redshift the maps were computed for (Lenstool ``dpl`` line).
    x_points_west : bool, default True
        True if the x map is the deflection along WEST (Lenstool convention);
        it is then negated so that alpha_x points east like theta_x. Set False
        for maps whose x deflection already points east.

    Returns
    -------
    DeflectionField

    Raises
    ------
    ValueError
        If the two maps have different shapes or WCS, the WCS is rotated, or
        the projection is not TAN.
    """
    alpha_x, hdr_x = _read_map(path_x)
    alpha_y, hdr_y = _read_map(path_y)
    if alpha_x.shape != alpha_y.shape:
        raise ValueError(f"map shapes differ: {alpha_x.shape} vs {alpha_y.shape}")

    wcs = WCS(hdr_x).celestial
    if not wcs.wcs.compare(WCS(hdr_y).celestial.wcs, tolerance=1e-10):
        raise ValueError("dplx and dply have different WCS")
    if not (wcs.wcs.ctype[0].endswith("TAN") and wcs.wcs.ctype[1].endswith("TAN")):
        raise ValueError(f"expected a TAN projection, got {list(wcs.wcs.ctype)}")

    # Use RA/Dec as given (no FK5 -> ICRS transformation); see module docstring.
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = np.nan

    # pixel_scale_matrix = CDELT * PC in deg/px; off-diagonal terms mean rotation.
    m = wcs.pixel_scale_matrix
    if not np.allclose([m[0, 1], m[1, 0]], 0.0, atol=1e-12):
        raise ValueError("rotated deflection-map WCS is not supported")

    if x_points_west:
        alpha_x = -alpha_x

    return DeflectionField(
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        wcs=wcs,
        step_x=float(m[0, 0]) * 3600.0,
        step_y=float(m[1, 1]) * 3600.0,
        i0=float(wcs.wcs.crpix[1]) - 1.0,
        j0=float(wcs.wcs.crpix[0]) - 1.0,
        z_ref=float(z_ref),
    )
