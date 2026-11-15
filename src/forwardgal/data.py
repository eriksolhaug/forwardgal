"""Loading imaging data, noise maps and cutouts.

Conventions
-----------
- Arrays are indexed ``[row, col]`` = ``[y, x]`` and are 0-based. FITS keywords
  such as CRPIX are 1-based; astropy's WCS handles that offset for us as long as
  we always call it with ``origin=0`` / the ``pixel_to_world`` API.
- Pixel values keep the units of the input file (e.g. MJy/sr for JWST i2d
  mosaics). Magnitudes are computed relative to a zeropoint, as in GALFIT.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


@dataclass
class ImageData:
    """A cutout of an image and its 1-sigma noise map, on the same pixel grid.

    Attributes
    ----------
    data : (Ny, Nx) float64 array
        Pixel values, in the units given by ``bunit``.
    sigma : (Ny, Nx) float64 array
        1-sigma uncertainty per pixel, same units as ``data``.
    wcs : astropy.wcs.WCS
        Celestial WCS of the cutout (maps 0-based [row, col] to RA, Dec).
    pixscale : float
        Pixel side length in arcsec (pixels are required to be square).
    bunit : str
        Value of the BUNIT keyword of the data extension (e.g. "MJy/sr").
    zeropoint : float or None
        Magnitude zeropoint: m = -2.5 log10(sum of pixel values) + zeropoint.
    """

    data: np.ndarray
    sigma: np.ndarray
    wcs: WCS
    pixscale: float
    bunit: str
    zeropoint: float | None


def read_image(path, ext):
    """Read one image extension from a FITS file.

    Parameters
    ----------
    path : str or Path
        FITS file.
    ext : int or str
        Extension number (0 = primary) or name (e.g. "SCI", "ERR").

    Returns
    -------
    array : (Ny, Nx) array
        The image, memory-mapped (not yet copied into RAM).
    header : astropy.io.fits.Header
        The header of that extension.
    """
    hdul = fits.open(path, memmap=True)
    hdu = hdul[ext]
    if hdu.data is None or hdu.data.ndim != 2:
        raise ValueError(f"{path}[{ext}] is not a 2-D image")
    return hdu.data, hdu.header


def pixel_scale_arcsec(wcs):
    """Pixel side length in arcsec, for a WCS with square pixels.

    Raises ValueError if the two axes have different scales (non-square pixels),
    since the rest of forwardgal assumes square pixels.
    """
    # proj_plane_pixel_scales gives the size of one pixel along each axis, in deg.
    scales = wcs.proj_plane_pixel_scales()
    sx, sy = (s.to_value(u.arcsec) for s in scales)
    if not np.isclose(sx, sy, rtol=1e-6):
        raise ValueError(f"non-square pixels: {sx} x {sy} arcsec")
    return float(sx)


def ab_zeropoint(header):
    """AB magnitude zeropoint for an image in MJy/sr.

    With f_pix the pixel values in MJy/sr and Omega_pix the pixel solid angle in
    steradian (header keyword PIXAR_SR), the total flux in Jy of a source is
    sum(f_pix) * 1e6 * Omega_pix, and the AB magnitude is
    m = -2.5 log10(flux_Jy / 3631). Writing m = -2.5 log10(sum f_pix) + ZP gives

        ZP = -2.5 log10(1e6 * Omega_pix / 3631).

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header of the data extension. Must contain BUNIT = "MJy/sr" and PIXAR_SR.

    Returns
    -------
    float
        The zeropoint (about 28.087 for NIRCam short-wavelength i2d mosaics at
        0.03"/px).

    Raises
    ------
    ValueError
        If BUNIT is not MJy/sr or PIXAR_SR is missing.
    """
    # TODO(Erik): implement this.
    #   1. Check header.get("BUNIT") == "MJy/sr"; otherwise raise ValueError
    #      with a helpful message (other units need a different formula).
    #   2. Read PIXAR_SR (raise ValueError if it is missing).
    #   3. Return the zeropoint from the formula in the docstring, as a float.
    # Check your version with:  pytest tests/test_data.py
    if header.get("BUNIT") != "MJy/sr":
        raise ValueError("Units of the header file are not in MJy/sr.")
    if "PIXAR_SR" not in header:
        raise ValueError("PIXAR_SR not in header.")
    else:
        zeropoint = float(-2.5*np.log10(1e6 * header.get("PIXAR_SR")/3631))
    return zeropoint


def load_cutout(
    data_path,
    data_ext,
    sigma_path,
    sigma_ext,
    ra_deg,
    dec_deg,
    size_arcsec,
    zeropoint=None,
):
    """Cut out matching square regions of a data image and its sigma map.

    Parameters
    ----------
    data_path, data_ext : str/Path, int/str
        File and extension of the science image (e.g. i2d file, "SCI").
    sigma_path, sigma_ext : str/Path, int/str
        File and extension of the 1-sigma map (e.g. same file, "ERR").
    ra_deg, dec_deg : float
        Centre of the cutout (ICRS degrees).
    size_arcsec : float
        Side length of the square cutout in arcsec. The pixel size is rounded
        to the nearest integer number of pixels.
    zeropoint : float, optional
        Magnitude zeropoint. If None, it is computed from the data header with
        ``ab_zeropoint`` (requires BUNIT = MJy/sr).

    Returns
    -------
    ImageData
        Arrays are float64 copies with shape (N, N), indexed [row, col].
    """
    data_full, data_hdr = read_image(data_path, data_ext)
    sigma_full, sigma_hdr = read_image(sigma_path, sigma_ext)

    if data_full.shape != sigma_full.shape:
        raise ValueError(
            f"data shape {data_full.shape} != sigma shape {sigma_full.shape}"
        )

    # WCS(header).celestial keeps only the RA/Dec axes.
    wcs_full = WCS(data_hdr).celestial
    # Error maps often carry no WCS of their own; only compare if one is present.
    if "CTYPE1" in sigma_hdr:
        sigma_wcs = WCS(sigma_hdr).celestial
        if not wcs_full.wcs.compare(sigma_wcs.wcs, tolerance=1e-10):
            raise ValueError("data and sigma images have different WCS")

    pixscale = pixel_scale_arcsec(wcs_full)
    npix = int(round(size_arcsec / pixscale))
    centre = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")

    # Cutout2D takes size as (ny, nx) in pixels and returns a cutout with a
    # correctly shifted WCS. mode="strict" fails if the box leaves the image.
    cut_data = Cutout2D(data_full, centre, (npix, npix), wcs=wcs_full, mode="strict")
    cut_sigma = Cutout2D(sigma_full, centre, (npix, npix), wcs=wcs_full, mode="strict")

    if zeropoint is None:
        zeropoint = ab_zeropoint(data_hdr)

    return ImageData(
        # astype(float64) also converts FITS big-endian bytes to native order.
        data=cut_data.data.astype(np.float64),
        sigma=cut_sigma.data.astype(np.float64),
        wcs=cut_data.wcs,
        pixscale=pixscale,
        bunit=str(data_hdr.get("BUNIT", "")),
        zeropoint=zeropoint,
    )
