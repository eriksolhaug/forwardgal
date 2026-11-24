"""Milestone 6a: load the PSF, convolve a lensed model, and check the margin.

1. PSF summary (GALFIT conventions: unit sum, centre pixel [N//2, N//2]).
2. Convolution of a lensed host model on the render box of an example mask,
   with timing (value and gradient).
3. The mask-edge effect: convolving the model rendered on the mask ONLY vs
   on the mask + 35 px margin, compared inside the mask in units of sigma.

Run from the repository root:
    python scripts/m6a_psf.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import forwardgal  # noqa: F401
from common import load_example_cutout, load_field, example_mask, PSF_FILE
from forwardgal.psf import load_psf, make_convolver
from forwardgal.regions import fit_region
from forwardgal.render import render_lensed_sersic
from forwardgal.traced import trace_grid


psf = load_psf(PSF_FILE)
print(psf.describe())

img = load_example_cutout()
field = load_field()
ny, nx = img.data.shape
region = fit_region(example_mask((ny, nx)), margin=35)
r0, r1, c0, c1 = region.box
grid = trace_grid(img.wcs, region.box_shape, field, subsample=5, origin=(r0, c0))
conv = make_convolver(psf.kernel, region.box_shape)
pix = region.render_pixels_in_box()
mask_box = region.in_box(region.mask)
sigma_box = region.in_box(img.sigma)

host = dict(x0=-5.728, y0=0.114, I_e=1.0, R_e=0.15, n=2.0, q=0.6, pa_deg=40.0)
model = render_lensed_sersic(grid, **host, pixels=pix)
scale = 0.5 / float(model.max())  # host peak 0.5 MJy/sr, as in M4c
model = model * scale
blurred = conv(model)
print(f"\nrender box {region.box_shape}: flux before / after convolution = "
      f"{float(blurred.sum() / model.sum()):.4f} (< 1: light spread outside the box)")

# Timing: value and gradient of render + convolve.
f = jax.jit(lambda p: conv(render_lensed_sersic(grid, p[0], p[1], 1.0, p[2], p[3], 0.6, 40.0, pixels=pix)))
g = jax.jit(jax.grad(lambda p: jnp.sum(f(p) ** 2)))
p0 = jnp.array([-5.728, 0.114, 0.15, 2.0])
f(p0).block_until_ready()
g(p0).block_until_ready()
t0 = time.perf_counter()
for _ in range(10):
    f(p0).block_until_ready()
tv = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for _ in range(10):
    g(p0).block_until_ready()
tg = (time.perf_counter() - t0) / 10
tc0 = time.perf_counter()
jc = jax.jit(conv)
jc(model).block_until_ready()
tc0 = time.perf_counter()
for _ in range(20):
    jc(model).block_until_ready()
tc = (time.perf_counter() - tc0) / 20
print(f"timing: convolution alone {1e3 * tc:.1f} ms; render + convolve {1e3 * tv:.1f} ms, gradient {1e3 * tg:.1f} ms")

# Mask-edge effect: render on the mask only (zero elsewhere) vs with the margin.
mrows, mcols = np.nonzero(mask_box)
model_mask_only = render_lensed_sersic(grid, **host, pixels=(mrows, mcols)) * scale
dev = np.abs(np.asarray(conv(model_mask_only) - blurred))[mask_box] / sigma_box[mask_box]
print(f"mask only vs mask + 35 px margin, inside the mask: max {dev.max():.2f} sigma, "
      f"median {np.median(dev):.3f} sigma")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, arr, title in [
    (axes[0], np.log10(psf.kernel), "PSF, log10 (normalized)"),
    (axes[1], np.asarray(blurred), "lensed host * PSF [MJy/sr]"),
    (axes[2], np.where(mask_box, np.asarray(blurred - conv(model_mask_only)) / sigma_box, np.nan),
     "inside mask: (with margin - mask only) / sigma"),
]:
    im = ax.imshow(arr, origin="lower", cmap="magma" if ax is not axes[2] else "viridis")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
out = Path("outputs/m6a_psf.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved figure: {out}")
plt.show()
