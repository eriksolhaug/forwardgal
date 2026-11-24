"""Lens distortion, for talks: the mask's pixel grid in both planes.

Left : data with every 6th pixel-grid line of the mask (image plane).
Right: the same lines traced to the source plane (beta = theta - alpha),
       over the source-plane model (host shape guessed in M8; I_e from the
       M8 amplitude solve), with caustics and the quasar.
The GUI will offer the traced grid as a toggle on the source-plane view.

Run from the repository root:
    python scripts/m8b_lens_distortion.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

import forwardgal  # noqa: F401
from common import QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.critical import caustics, critical_curves
from forwardgal.plotting import plot_traced_grid
from forwardgal.quasar import image_fixed_quasar
from forwardgal.regions import fit_region
from forwardgal.render import source_plane_image
from forwardgal.traced import trace_grid

EVERY = 6  # draw every 6th grid line (0.18" apart in the image plane)

img = load_example_cutout()
field = load_field()
region = fit_region(example_mask(img.data.shape), margin=35)
r0, r1, c0, c1 = region.box
grid = trace_grid(img.wcs, region.box_shape, field, subsample=1, origin=(r0, c0))
mask = region.in_box(region.mask)
qso = image_fixed_quasar(*QSO_1A, field)
host = dict(x0=qso.beta[0] + 0.01, y0=qso.beta[1] - 0.01, I_e=0.568, R_e=0.12, n=1.0, q=0.6, pa_deg=30.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))

# Image plane: data in theta coordinates, with the mask grid.
data = region.in_box(img.data)
tx, ty = grid.theta_corner_x, grid.theta_corner_y
ext_img = (tx[0, 0], tx[0, -1], ty[0, 0], ty[-1, 0])  # corners -> (left, right, bottom, top)
ax1.imshow(data, origin="lower", extent=ext_img, cmap="gray",
           norm=ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch()))
ROW_KW, COL_KW = dict(color="deepskyblue", lw=1.0), dict(color="orange", lw=1.0)
plot_traced_grid(ax1, grid, mask, every=EVERY, plane="image", row_kw=ROW_KW, col_kw=COL_KW)
ax1.contour(mask, levels=[0.5], colors="w", linewidths=1.2, linestyles="--", extent=ext_img, origin="lower")
ax1.plot(*qso.theta, "*", color="lime", ms=12, label="quasar 1A")
ax1.set_xlim(ext_img[0], ext_img[1])
ax1.set_xlabel("theta_x [arcsec, +east]")
ax1.set_ylabel("theta_y [arcsec, +north]")
ax1.set_title("image plane: data + mask pixel grid", fontsize=10)
ax1.legend(loc="lower left", fontsize=8)
ax1.set_aspect("equal")

# Source plane: model, traced grid, caustics, quasar.
src, ext_src = source_plane_image([host], centre=qso.beta, size=2.0, pixscale=0.005)
ax2.imshow(np.log10(np.maximum(src, 1e-3 * src.max())), origin="lower", extent=ext_src, cmap="magma")
plot_traced_grid(ax2, grid, mask, every=EVERY, plane="source", row_kw=ROW_KW, col_kw=COL_KW)
for k, c in enumerate(caustics(field, critical_curves(field))):
    ax2.plot(c[:, 0], c[:, 1], "w-", lw=1.0, label="caustics" if k == 0 else None)
ax2.plot(*qso.beta, "*", color="lime", ms=12, label="quasar (source position)")
ax2.set_xlim(ext_src[0], ext_src[1])
ax2.set_ylim(ext_src[2], ext_src[3])
ax2.set_xlabel("beta_x [arcsec, +east]")
ax2.set_ylabel("beta_y [arcsec, +north]")
ax2.set_title("source plane: the same grid lines, traced (beta = theta - alpha)", fontsize=10)
ax2.legend(loc="lower left", fontsize=8)
ax2.set_aspect("equal")

out = Path("outputs/m8b_lens_distortion.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved figure: {out}")
plt.show()
