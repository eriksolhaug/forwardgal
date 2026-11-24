"""Milestone 4b: lensed Sersic models on the real cutout around quasar image 1A.

1. A tiny source at the quasar's source position: its lensed flux divided by
   its intrinsic flux must equal the magnification of image 1A (the other
   images 1B, 1C lie outside this cutout).
2. A host-like source: hybrid renderer vs exact integration of every pixel,
   and timings.

Run from the repository root:
    python scripts/m4b_lensed_sersic.py
"""

import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

import forwardgal  # noqa: F401
from common import QSO_1A, load_example_cutout, load_field
from forwardgal.quasar import image_fixed_quasar
from forwardgal.profiles.sersic import sersic_total_flux
from forwardgal.render import render_lensed_sersic, render_lensed_sersic_exact
from forwardgal.traced import trace_grid

img = load_example_cutout()
field = load_field()
grid = trace_grid(img.wcs, img.data.shape, field, subsample=5)
pix_area = img.pixscale**2
BETA_QSO = image_fixed_quasar(*QSO_1A, field).beta  # PSF-fitted 1A traced back (M6b)

# 1. Magnification of a tiny source at the quasar position.
R_e, n, q = 0.003, 1.0, 1.0
tiny = np.asarray(render_lensed_sersic(grid, *BETA_QSO, 1.0, R_e, n, q, 0.0))
mu_flux = tiny.sum() * pix_area / float(sersic_total_flux(1.0, R_e, n, q))
mu_area = 1.0 / grid.area_ratio()
r, c = np.unravel_index(tiny.argmax(), tiny.shape)
print(f"tiny source at beta_qso: lensed/intrinsic flux = {mu_flux:.3f}")
print(f"  footprint magnification at its image pixel [{r}, {c}]: {mu_area[r, c]:.3f}  (Lenstool at 1A: 5.418)")

# 2. A host-like source, slightly offset from the quasar.
host = dict(x0=BETA_QSO[0] + 0.02, y0=BETA_QSO[1] - 0.01, I_e=1.0, R_e=0.15, n=2.0, q=0.6, pa_deg=40.0)
# Parameters are passed as arguments (as in a fit): otherwise JAX treats
# everything as a constant and tries to pre-compute it at compile time.
p_host = np.array([host[k] for k in ("x0", "y0", "I_e", "R_e", "n", "q", "pa_deg")])
f_h = jax.jit(lambda p: render_lensed_sersic(grid, *p))
f_e = jax.jit(lambda p: render_lensed_sersic_exact(grid, *p))
hybrid = np.asarray(f_h(p_host))
t0 = time.perf_counter()
for _ in range(10):
    f_h(p_host).block_until_ready()
t_h = (time.perf_counter() - t0) / 10
exact = np.asarray(f_e(p_host))
t0 = time.perf_counter()
f_e(p_host).block_until_ready()
t_e = time.perf_counter() - t0
diff = (hybrid - exact) / exact.max()
print(f"\nhost-like source (R_e = 0.15\", n = 2): max |hybrid - exact| / peak = {np.abs(diff).max():.1e}")
print(f"  render time: hybrid {1e3 * t_h:.1f} ms, exact everywhere {1e3 * t_e:.0f} ms  ({img.data.shape[0]}x{img.data.shape[1]} px)")

fig = plt.figure(figsize=(14, 4.6))
panels = [
    (img.data, "data (F115W)", ImageNormalize(img.data, interval=ZScaleInterval(), stretch=AsinhStretch()), "gray"),
    (np.log10(hybrid / hybrid.max()), "lensed host model, log10(I / I_max)", None, "magma"),
    (diff, "(hybrid - exact) / peak", None, "RdBu_r"),
]
for k, (arr, title, norm, cmap) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1, projection=img.wcs)
    kw = dict(norm=norm) if norm is not None else dict(vmin=-3, vmax=0) if k == 1 else dict(vmin=-1e-3, vmax=1e-3)
    im = ax.imshow(arr, origin="lower", cmap=cmap, **kw)
    ax.set_title(title, fontsize=9)
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    fig.colorbar(im, ax=ax, fraction=0.046)
out = Path("outputs/m4b_lensed_sersic.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved figure: {out}")
plt.show()
