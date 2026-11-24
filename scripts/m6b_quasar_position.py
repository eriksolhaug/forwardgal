"""Milestone 6b: measure quasar image 1A's position by fitting the PSF to the data.

Model inside a circle of radius rho around the quasar peak:
    data ~ flux * PSF(shifted by (sy, sx)) + background,
with sigma from the ERR map. flux and background are solved linearly for each
trial shift; (sy, sx) minimizes chi^2. The PSF stamp's centre pixel [70, 70]
is placed on the peak pixel and then shifted by (sy, sx).

The quasar position is the stamp's light centroid [69.829, 70.255] (where the
PSF was made to have the quasar) carried to the mosaic. If the PSF was made
for exactly this quasar, the best (sy, sx) should be close to whole pixels,
i.e. the residual fractional shift should be ~0.

Several radii are used: small radii are dominated by the quasar core, larger
ones include more host light.

Run from the repository root:
    python scripts/m6b_quasar_position.py
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize

import forwardgal  # noqa: F401
from forwardgal.lens import load_deflection
from forwardgal.psf import fourier_shift, load_psf

IMAGE = Path.home() / "Research/Data_and_Analysis/DATA/CJ1153/jw06675-o001_t002_nircam_clear-f115w_i2d.fits"
psf = load_psf("data/rotated_jw06675-o001_t002_nircam_clear-f115w-QSOblueA_PSF_i2dtrim.fits")
K = psf.kernel
kc = psf.centre[0]  # 70
yy, xx = np.indices(K.shape)
centroid = np.array([(K * yy).sum(), (K * xx).sum()])  # [69.829, 70.255]

with fits.open(IMAGE, memmap=True) as hdul:
    wcs = WCS(hdul["SCI"].header).celestial
    PR, PC = 3636, 1073  # brightest pixel near 1A (0-based [row, col]) from M6 check
    H = 20
    data = np.asarray(hdul["SCI"].data[PR - H : PR + H + 1, PC - H : PC + H + 1], float)
    sigma = np.asarray(hdul["ERR"].data[PR - H : PR + H + 1, PC - H : PC + H + 1], float)

rr, cc = np.indices(data.shape) - H


def model_unit(sy, sx):
    """PSF with its centre on the local centre, shifted by (sy, sx); cropped to the data stamp."""
    shifted = np.asarray(fourier_shift(K, sy, sx))
    return shifted[kc - H : kc + H + 1, kc - H : kc + H + 1]


def chi2(shift, inside):
    m = model_unit(*shift)[inside]
    d, s = data[inside], sigma[inside]
    A = np.column_stack([m / s, 1.0 / s])  # columns: flux, background
    coef, *_ = np.linalg.lstsq(A, d / s, rcond=None)
    return float(np.sum((d / s - A @ coef) ** 2)), coef


field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits", "data/lensmodel/bestopt_1.524_dply.fits", z_ref=1.524
)
ARCS_1A = (178.3352456, 7.9335292)

print(f"PSF light centroid in stamp: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
print(f"\n{'radius':>6s} {'npix':>5s} {'shift (rows, cols)':>20s} {'chi2/dof':>9s} {'quasar [row, col] in mosaic':>29s} {'RA, Dec':>28s}")
results = []
for rho in (1.5, 2.0, 3.0, 4.0, 6.0):
    inside = np.hypot(rr, cc) <= rho
    # coarse grid over +-1 px, then refine
    grid = np.linspace(-1.0, 1.0, 21)
    best = min(((sy, sx) for sy in grid for sx in grid), key=lambda s: chi2(s, inside)[0])
    res = minimize(lambda s: chi2(s, inside)[0], best, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6))
    sy, sx = res.x
    c2, (flux, bkg) = chi2(res.x, inside)
    dof = inside.sum() - 4
    # stamp point `centroid` lands at local (H + centroid - kc + shift) -> mosaic:
    q_row = PR - H + (H + centroid[0] - kc + sy)
    q_col = PC - H + (H + centroid[1] - kc + sx)
    ra, dec = wcs.wcs_pix2world(q_col, q_row, 0)
    results.append((rho, sy, sx, q_row, q_col, float(ra), float(dec), flux))
    print(f"{rho:6.1f} {inside.sum():5d} {f'({sy:+.3f}, {sx:+.3f})':>20s} {c2 / dof:9.2f} "
          f"{f'[{q_row:.3f}, {q_col:.3f}]':>29s} {f'{float(ra):.7f}, {float(dec):.7f}':>28s}")

best = results[1]  # rho = 2 px: core-dominated
_, sy, sx, q_row, q_col, ra, dec, flux = best
print(f"\nadopted (rho = 2 px): quasar at mosaic [row, col] = [{q_row:.3f}, {q_col:.3f}], RA, Dec = {ra:.7f}, {dec:.7f}")
print(f"residual fractional shift relative to whole-pixel placement: ({sy:+.3f}, {sx:+.3f}) px")
spread = np.ptp([r[3] for r in results[:3]]), np.ptp([r[4] for r in results[:3]])
print(f"spread over radii 1.5-3 px: {spread[0]:.3f} rows, {spread[1]:.3f} cols")

# Offset from the arcs-file position, and what it does to the back-traced beta.
c_arcs, r_arcs = wcs.wcs_world2pix(*ARCS_1A, 0)
print(f"\narcs-file 1A is at mosaic [{float(r_arcs):.3f}, {float(c_arcs):.3f}]: offset from the fitted position "
      f"{np.hypot(float(r_arcs) - q_row, float(c_arcs) - q_col) * 0.03:.3f}\"")
for label, (a, d) in [("fitted", (ra, dec)), ("arcs file", ARCS_1A)]:
    tx, ty = field.sky_to_theta(a, d)
    bx, by = field.backtrace(tx, ty)
    print(f"  beta from {label:9s} position: ({float(bx):+.4f}, {float(by):+.4f}) arcsec")
