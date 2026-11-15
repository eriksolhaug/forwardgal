"""Milestone 1: load a 5" x 5" cutout around quasar image 1A and plot it.

Run from the repository root:
    python scripts/m1_cutout.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

from forwardgal.data import load_cutout

IMAGE = Path.home() / "Research/Data_and_Analysis/DATA/CJ1153/jw06675-o001_t002_nircam_clear-f115w_i2d.fits"
RA_1A, DEC_1A = 178.3352456, 7.9335292  # quasar image 1A (arcs/1153arc_v2_zvary.dat)

img = load_cutout(IMAGE, "SCI", IMAGE, "ERR", RA_1A, DEC_1A, size_arcsec=5.0)

finite = np.isfinite(img.data) & np.isfinite(img.sigma)
print(f"cutout shape      : {img.data.shape}  ([row, col] = [y, x])")
print(f"pixel scale       : {img.pixscale:.4f} arcsec/px")
print(f"units (BUNIT)     : {img.bunit}")
print(f"zeropoint (AB)    : {img.zeropoint:.3f}")
print(f"non-finite pixels : {(~finite).sum()}")
print(f"data  min/med/max : {np.nanmin(img.data):.3g} / {np.nanmedian(img.data):.3g} / {np.nanmax(img.data):.3g}")
print(f"sigma min/med/max : {np.nanmin(img.sigma):.3g} / {np.nanmedian(img.sigma):.3g} / {np.nanmax(img.sigma):.3g}")
print(f"total flux in box : m_AB = {-2.5 * np.log10(np.nansum(img.data)) + img.zeropoint:.2f}")

fig = plt.figure(figsize=(13, 5))
fig.subplots_adjust(wspace=0.45)
for i, (arr, title) in enumerate([(img.data, "data (SCI)"), (img.sigma, "sigma (ERR)")]):
    ax = fig.add_subplot(1, 2, i + 1, projection=img.wcs)
    norm = ImageNormalize(arr, interval=ZScaleInterval(), stretch=AsinhStretch())
    im = ax.imshow(arr, origin="lower", cmap="gray", norm=norm)
    ax.plot(RA_1A, DEC_1A, "r+", ms=14, transform=ax.get_transform("world"))
    ax.set_title(f"{title} [{img.bunit}]")
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_major_formatter("hh:mm:ss.ss")
    ax.coords[1].set_major_formatter("dd:mm:ss.s")
    fig.colorbar(im, ax=ax, fraction=0.046)

out = Path("outputs/m1_cutout.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved figure      : {out}")
plt.show()
