"""Milestone 7b: source-fixed and free quasar modes on the real data.

Source position: beta from the PSF-fitted JWST position of 1A (M6b).
For each quasar image (1A, 1B, 1C), on a 41 x 41 px region of the F115W mosaic
around its observed peak:
  - source-fixed: where the lens model puts the image;
  - free mode (beta held fixed, per-image offset free): fit the offset, the
    flux and a local background to the data within 3 px of the peak, using
    the 1A PSF everywhere.
The fitted offset is the lens model's error at that image; chi2/dof shows how
well the 1A PSF describes each image.

Run from the repository root:
    python scripts/m7b_quasar_modes.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize

import forwardgal  # noqa: F401
from forwardgal.lens import load_deflection
from forwardgal.psf import load_psf, make_point_source_renderer
from forwardgal.quasar import free_quasar, grid_mapping, image_fixed_quasar, source_fixed_quasar

IMAGE = Path.home() / "Research/Data_and_Analysis/DATA/CJ1153/jw06675-o001_t002_nircam_clear-f115w_i2d.fits"
field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits", "data/lensmodel/bestopt_1.524_dply.fits", z_ref=1.524
)
psf = load_psf("data/rotated_jw06675-o001_t002_nircam_clear-f115w-QSOblueA_PSF_i2dtrim.fits")
yy, xx = np.indices(psf.kernel.shape)
PSF_REF = ((psf.kernel * yy).sum(), (psf.kernel * xx).sum())  # quasar position in the stamp

beta = image_fixed_quasar(178.3352306, 7.9335254, field).beta
print(f"source position from the PSF-fitted 1A: beta = ({beta[0]:+.4f}, {beta[1]:+.4f})\"\n")
OBS = {"1A": (178.3352306, 7.9335254), "1B": (178.3317916, 7.9364979), "1C": (178.3289710, 7.9369786)}

H, SHAPE = 20, (41, 41)
with fits.open(IMAGE, memmap=True) as hdul:
    wcs = WCS(hdul["SCI"].header).celestial
    sci, err = hdul["SCI"].data, hdul["ERR"].data
    print(f"{'image':5s} {'source-fixed miss':>18s} {'fitted offset (E, N)':>24s} {'|offset|':>9s} {'chi2/dof':>9s}")
    for name, (ra, dec) in OBS.items():
        col, row = wcs.wcs_world2pix(ra, dec, 0)
        r0, c0 = int(round(float(row))), int(round(float(col)))
        sub = np.asarray(sci[r0 - 4 : r0 + 5, c0 - 4 : c0 + 5], float)
        pk = np.unravel_index(np.nanargmax(sub), sub.shape)
        PR, PC = r0 - 4 + pk[0], c0 - 4 + pk[1]  # brightest pixel = data peak
        data = np.asarray(sci[PR - H : PR + H + 1, PC - H : PC + H + 1], float)
        sigma = np.asarray(err[PR - H : PR + H + 1, PC - H : PC + H + 1], float)
        mapping = grid_mapping(wcs, field, origin=(PR - H, PC - H), shape=SHAPE)

        sf = source_fixed_quasar(*beta, field, mapping, SHAPE, pad=0)
        if not sf.in_region:
            print(f"{name:5s}  no predicted image within this region")
            continue
        pr, pc = sf.grid_positions[0]
        miss = np.hypot(pr - H, pc - H) * 0.03  # predicted position vs data peak

        fq = free_quasar(*beta, field, mapping, SHAPE, pad=0)
        render = make_point_source_renderer(psf.kernel, SHAPE, PSF_REF)
        rr, cc = np.indices(SHAPE) - H
        inside = np.hypot(rr, cc) <= 3.0
        d, s = jnp.asarray(data[inside]), jnp.asarray(sigma[inside])

        @jax.jit
        def chi2(offset):
            m = fq.unit_images(jnp.asarray(beta), offset.reshape(1, 2), render)[0][inside]
            A = jnp.stack([m / s, 1.0 / s], axis=1)  # flux, background (linear)
            coef = jnp.linalg.lstsq(A, d / s)[0]
            return jnp.sum((d / s - A @ coef) ** 2)

        grad = jax.jit(jax.grad(chi2))
        # Start from the data peak: offset = M^-1 (peak - predicted) in theta.
        start = np.linalg.solve(mapping.M, np.array([H - pr, H - pc]))
        res = minimize(lambda o: float(chi2(jnp.asarray(o))), start,
                       jac=lambda o: np.asarray(grad(jnp.asarray(o))), method="BFGS")
        off = res.x
        print(f"{name:5s} {miss:17.3f}\" {f'({off[0]:+.3f}, {off[1]:+.3f})':>23s}\" {np.hypot(*off):8.3f}\" "
              f"{res.fun / (inside.sum() - 4):9.1f}")
