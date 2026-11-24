"""Milestone 4a: trace the cutout to the source plane and check magnifications.

Compares 1 / (footprint area / pixel area) of every traced pixel with
Lenstool's own magnification map (bestopt_1.524_magn.fits, which holds the
signed magnification mu), and shows where the parity flips (mu < 0), i.e.
where a critical curve crosses the cutout.

Run from the repository root:
    python scripts/m4a_traced_grid.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import forwardgal  # noqa: F401
from common import load_example_cutout, load_field, LENSTOOL_MODEL
from forwardgal.traced import trace_grid

MAGN = LENSTOOL_MODEL / "fits/bestopt_1.524_magn.fits"

img = load_example_cutout()
field = load_field()
grid = trace_grid(img.wcs, img.data.shape, field, subsample=5)
mu = 1.0 / grid.area_ratio()

# Lenstool's mu at each cutout pixel centre (nearest deflection-map pixel).
ny, nx = img.data.shape
rows, cols = np.mgrid[0:ny, 0:nx]
ra, dec = img.wcs.wcs_pix2world(cols.ravel(), rows.ravel(), 0)
mr, mc = field.theta_to_pixel(*field.sky_to_theta(ra, dec))
mu_lt = fits.getdata(MAGN).astype(float)[np.round(mr).astype(int), np.round(mc).astype(int)].reshape(ny, nx)

ratio = mu_lt / mu
c1a = (img.data.shape[0] // 2, img.data.shape[1] // 2)  # cutout centred on 1A
print(f"mu at 1A (pixel {list(c1a)}) : ours {mu[c1a]:.3f}, Lenstool {mu_lt[c1a]:.3f}")
print(f"Lenstool mu / our mu         : median {np.median(ratio):.4f}, 5-95% {np.percentile(ratio, 5):.4f} .. {np.percentile(ratio, 95):.4f}")
print(f"pixels with mu < 0 (flipped) : {(mu < 0).sum()} of {mu.size}")
bx, by = grid.beta_corner_x, grid.beta_corner_y
print(f"source-plane extent of cutout: beta_x {bx.min():+.2f} .. {bx.max():+.2f}, beta_y {by.min():+.2f} .. {by.max():+.2f} arcsec")

fig = plt.figure(figsize=(13, 4.6))
ax1 = fig.add_subplot(1, 3, 1, projection=img.wcs)
im = ax1.imshow(np.sign(mu) * np.log10(np.abs(mu)), origin="lower", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
ax1.contour(mu, levels=[0], colors="k", linewidths=1)
ax1.set_title("sign(mu) log10|mu|  (black: mu = 0 boundary)", fontsize=9)
fig.colorbar(im, ax=ax1, fraction=0.046)
ax2 = fig.add_subplot(1, 3, 2, projection=img.wcs)
im = ax2.imshow(ratio, origin="lower", cmap="coolwarm", vmin=0.95, vmax=1.05)
ax2.set_title("Lenstool mu / our mu", fontsize=9)
fig.colorbar(im, ax=ax2, fraction=0.046)
ax3 = fig.add_subplot(1, 3, 3)
# Every 6th row/column of the traced corner grid, drawn in the source plane.
for i in range(0, bx.shape[0], 6):
    ax3.plot(bx[i], by[i], "C0", lw=0.5)
for j in range(0, bx.shape[1], 6):
    ax3.plot(bx[:, j], by[:, j], "C1", lw=0.5)
ax3.set_xlim(ax3.get_xlim()[::-1])  # east (+beta_x) to the left
ax3.set_aspect("equal")
ax3.set_xlabel("beta_x [arcsec, +east]")
ax3.set_ylabel("beta_y [arcsec, +north]")
ax3.set_title("cutout pixel grid traced to the source plane", fontsize=9)
for ax in (ax1, ax2):
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
out = Path("outputs/m4a_traced_grid.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved figure                 : {out}")
plt.show()
