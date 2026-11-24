"""Fit region on the real cutout, and the speed of rendering only what is needed.

Uses the example mask from scripts/common.py (ellipse aligned with the arc
around quasar image 1A; the GUI will let you draw the real one), shows the fit pixels, render
pixels and render box, and times one Sersic component (value and gradient)
rendered on: the whole cutout, the render box, and the render pixels only.

Run from the repository root:
    python scripts/m3c_region_benchmark.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from common import example_mask, load_example_cutout
from forwardgal.regions import fit_region
from forwardgal.render import render_sersic

img = load_example_cutout()
ny, nx = img.data.shape

# Example mask: ellipse centred on the cutout centre, long axis ~north-south.
region = fit_region(example_mask((ny, nx)), margin=35)
print(region.describe())

# --- timing: one Sersic, theta = 0 at the cutout centre ----------------------
row0, col0 = ny // 2, nx // 2
r0, r1, c0, c1 = region.box
pix = region.render_pixels_in_box()
kw = dict(pixscale=img.pixscale, I_e=1.0, q=0.6, pa_deg=20.0)
variants = {
    "whole cutout": lambda p: render_sersic((ny, nx), row0=row0, col0=col0, x0=p[0], y0=p[1], R_e=p[2], n=p[3], **kw),
    "render box": lambda p: render_sersic(region.box_shape, row0=row0 - r0, col0=col0 - c0,
                                          x0=p[0], y0=p[1], R_e=p[2], n=p[3], **kw),
    "render pixels": lambda p: render_sersic(region.box_shape, row0=row0 - r0, col0=col0 - c0,
                                             x0=p[0], y0=p[1], R_e=p[2], n=p[3], pixels=pix, **kw),
}
p0 = jnp.array([0.01, -0.02, 0.15, 2.5])
print(f"\n{'variant':14s} {'pixels':>8s} {'value [ms]':>11s} {'gradient [ms]':>14s}")
ref = None
for name, fn in variants.items():
    f = jax.jit(fn)
    g = jax.jit(jax.grad(lambda p: jnp.sum(fn(p) ** 2)))
    out = f(p0).block_until_ready()
    g(p0).block_until_ready()  # compile first, then time
    t0 = time.perf_counter()
    for _ in range(20):
        f(p0).block_until_ready()
    tv = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        g(p0).block_until_ready()
    tg = (time.perf_counter() - t0) / 20
    npx = {"whole cutout": ny * nx, "render box": out.size, "render pixels": len(pix[0])}[name]
    print(f"{name:14s} {npx:8d} {1e3 * tv:11.2f} {1e3 * tg:14.2f}")
    # Same values on the fit pixels in every variant:
    on_mask = np.asarray(out)[region.in_box(region.mask)] if name != "whole cutout" else np.asarray(out)[region.mask]
    ref = on_mask if ref is None else ref
    assert np.allclose(on_mask, ref, rtol=1e-12), name

# --- picture ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection=img.wcs))
norm = ImageNormalize(img.data, interval=ZScaleInterval(), stretch=AsinhStretch())
ax.imshow(img.data, origin="lower", cmap="gray", norm=norm)
ax.contour(region.render, levels=[0.5], colors="C1", linewidths=1.2)
ax.contour(region.mask, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")
ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec="C0", lw=1.5))
ax.set_title("white dashed: fit pixels (mask)\norange: render pixels (+35 px)   blue: render box", fontsize=9)
ax.coords[0].set_axislabel("RA")
ax.coords[1].set_axislabel("Dec")
out = Path("outputs/m3c_region.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved figure: {out}")
plt.show()
