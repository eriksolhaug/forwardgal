"""Milestone 4c: does the interpolation of the deflection maps matter?

Compares bilinear interpolation of alpha (current default) with a bicubic
spline: (1) traced positions, (2) magnification against Lenstool's own map
(bestopt_1.524_magn.fits, signed mu), (3) lensed model images in units of
the noise sigma, with and without PSF convolution. Host brightness is scaled
to a peak of 0.5 MJy/sr (a little brighter than the arc in the data).

Run from the repository root:
    python scripts/m4c_alpha_interpolation.py
"""

import numpy as np, forwardgal, dataclasses
from common import load_example_cutout, load_field, QSO_1A, LENSTOOL_MODEL
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from scipy.signal import fftconvolve
from forwardgal.lens import load_deflection
from forwardgal.data import load_cutout
from forwardgal.traced import trace_grid
from forwardgal.render import render_lensed_sersic

# Baseline: bilinear interpolation (bicubic is the package default since M4c).
field = load_field(interpolation="bilinear")
img = load_example_cutout()
ny, nx = field.alpha_x.shape
rows, cols = np.arange(ny), np.arange(nx)
sx = RectBivariateSpline(rows, cols, field.alpha_x, kx=3, ky=3)
sy = RectBivariateSpline(rows, cols, field.alpha_y, kx=3, ky=3)

class Bicubic:
    """Same as DeflectionField but alpha from a bicubic spline (for this test only)."""
    def __init__(self, f): self.f = f
    def sky_to_theta(self, ra, dec): return self.f.sky_to_theta(ra, dec)
    def backtrace(self, tx, ty):
        r, c = self.f.theta_to_pixel(tx, ty)
        r, c = np.asarray(r), np.asarray(c)
        return tx - sx(r, c, grid=False), ty - sy(r, c, grid=False)

g_lin = trace_grid(img.wcs, img.data.shape, field, 5)
g_cub = trace_grid(img.wcs, img.data.shape, Bicubic(field), 5)
print("max |beta_bilinear - beta_bicubic| over subpixels: %.2e arcsec" % max(
    np.abs(g_lin.beta_sub_x - g_cub.beta_sub_x).max(), np.abs(g_lin.beta_sub_y - g_cub.beta_sub_y).max()))
mu_lin, mu_cub = 1 / g_lin.area_ratio(), 1 / g_cub.area_ratio()
magn = fits.getdata(LENSTOOL_MODEL / "fits/bestopt_1.524_magn.fits").astype(float)
from scipy.interpolate import RegularGridInterpolator
from forwardgal.traced import pixel_theta
pr, pc = np.mgrid[0:img.data.shape[0], 0:img.data.shape[1]]
tx, ty = pixel_theta(img.wcs, field, pr, pc); mr, mc = field.theta_to_pixel(tx, ty)
mu_lt = RegularGridInterpolator((rows, cols), magn)(np.column_stack([np.ravel(mr), np.ravel(mc)])).reshape(img.data.shape)
for name, mu in [("bilinear", mu_lin), ("bicubic", mu_cub)]:
    r = mu_lt / mu
    print(f"Lenstool mu / {name:8s} mu: 5-95% {np.percentile(r,5):.4f} .. {np.percentile(r,95):.4f}")

psf = fits.getdata("data/rotated_jw06675-o001_t002_nircam_clear-f115w-QSOblueA_PSF_i2dtrim.fits").astype(float); psf /= psf.sum()
sigma = img.sigma
from forwardgal.quasar import image_fixed_quasar
BQ = image_fixed_quasar(*QSO_1A, field).beta
print("\nmax |model_bilinear - model_bicubic| / sigma over the cutout (host peak scaled to 0.5 MJy/sr)")
print(f"{'source':34s} {'unconvolved':>12s} {'PSF-convolved':>14s}")
for label, p in [("extended disk  R_e=0.30\" n=1", dict(R_e=0.30, n=1.0, q=0.6, pa_deg=40.0)),
                 ("bulge          R_e=0.08\" n=4", dict(R_e=0.08, n=4.0, q=0.7, pa_deg=-25.0)),
                 ("clump          R_e=0.03\" n=0.6", dict(R_e=0.03, n=0.6, q=0.8, pa_deg=10.0))]:
    a = np.asarray(render_lensed_sersic(g_lin, BQ[0] + 0.02, BQ[1] - 0.01, 1.0, **p))
    b = np.asarray(render_lensed_sersic(g_cub, BQ[0] + 0.02, BQ[1] - 0.01, 1.0, **p))
    scale = 0.5 / a.max()
    d_raw = np.abs(a - b) * scale / sigma
    d_psf = np.abs(fftconvolve(a, psf, mode="same") - fftconvolve(b, psf, mode="same")) * scale / sigma
    print(f"{label:34s} {d_raw.max():12.3f} {d_psf.max():14.3f}")
