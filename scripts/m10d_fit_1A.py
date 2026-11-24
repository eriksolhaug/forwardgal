"""Milestone 10d: the first real fit of the quasar 1A host (F115W).

Model: lensed host Sersic (source plane) + image-fixed quasar 1A (its own PSF,
placed with no shift) + sky, in the example mask (scripts/common.py).
5 L-BFGS starts from a rough guess; Laplace errors; rendering check:
  (A) rendering accuracy: hybrid host vs exact integration of every render
      pixel, both PSF-convolved, compared in the mask;
  (B) margin: exact host on the render pixels only vs on the whole render box
      (light from beyond the 35 px margin that reaches the mask).
Writes outputs/m10d_fit_1A.png.

Run from the repository root (behind a memory guard if unsure):
    python scripts/m10d_fit_1A.py
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import forwardgal  # noqa: F401
from common import PSF_FILE, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.diagnostics import rendering_check
from forwardgal.fit import fit, laplace_errors
from forwardgal.model import ForwardModel
from forwardgal.plotting import plot_fit_summary
from forwardgal.profiles.sersic import unit_flux_ie
from forwardgal.psf import load_psf, make_convolver, make_point_source_renderer
from forwardgal.quasar import image_fixed_quasar, snap_to_psf
from forwardgal.regions import fit_region
from forwardgal.render import render_lensed_sersic, render_lensed_sersic_exact
from forwardgal.traced import trace_grid

img = load_example_cutout()
field = load_field()
region = fit_region(example_mask(img.data.shape), margin=35)
print(region.describe())
r0, r1, c0, c1 = region.box
shape = region.box_shape
grid = trace_grid(img.wcs, shape, field, subsample=5, origin=(r0, c0))
psf = load_psf(PSF_FILE)
yy, xx = np.indices(psf.kernel.shape)
psf_ref = ((psf.kernel * yy).sum(), (psf.kernel * xx).sum())
qso = image_fixed_quasar(*QSO_1A, field)
qso_rc, _ = snap_to_psf(qso.grid_position(img.wcs, origin=(r0, c0)), psf_ref, psf.centre)
qso_template = np.asarray(make_point_source_renderer(psf.kernel, shape, psf_ref)(*qso_rc))
convolve = make_convolver(psf.kernel, shape)
mask = region.in_box(region.mask)
data, sigma = region.in_box(img.data), region.in_box(img.sigma)
pixels = region.render_pixels_in_box()

start = dict(x0=qso.beta[0] + 0.01, y0=qso.beta[1] - 0.01, R_e=0.12, n=1.0, q=0.6, pa_deg=30.0)  # M8 guess
params = ModelParams([sersic_component("host", "source", **start), sky_component()])
model = ForwardModel(params, grid, convolve, data, sigma, mask, pixels, img.pixscale,
                     fixed_templates={"QSO 1A": qso_template})

result = fit(model, n_starts=5, seed=0)
print("\n" + result.describe(model))
lap = laplace_errors(model, result)
print("\n" + lap.describe(zeropoint=img.zeropoint))

# --- rendering check -----------------------------------------------------------
v = {k: float(x) for k, x in result.values["host"].items()}
ie = result.amplitudes["host"] * float(unit_flux_ie(v["R_e"], v["n"], v["q"], img.pixscale))
args = (v["x0"], v["y0"], ie, v["R_e"], v["n"], v["q"], v["pa_deg"])
hybrid = np.asarray(render_lensed_sersic(grid, *args, pixels=pixels))
exact_all = np.asarray(render_lensed_sersic_exact(grid, *args))
render_mask = np.zeros(shape, bool)
render_mask[pixels] = True
exact_rp = np.where(render_mask, exact_all, 0.0)
print("\nrendering check (A) hybrid vs exact, on the render pixels:")
chk_a = rendering_check(hybrid, exact_rp, data, sigma, mask, convolve=convolve)
print(f"   max |diff| / sigma = {chk_a['max_dev_sigma']:.4f}, delta chi2 = {chk_a['delta_chi2']:+.3f}, ok = {chk_a['ok']}")
print("rendering check (B) margin: render pixels only vs whole render box (both exact):")
chk_b = rendering_check(exact_rp, exact_all, data, sigma, mask, convolve=convolve)
print(f"   max |diff| / sigma = {chk_b['max_dev_sigma']:.4f}, delta chi2 = {chk_b['delta_chi2']:+.3f}, ok = {chk_b['ok']}")

check_text = (f"rendering check (A) hybrid vs exact: max {chk_a['max_dev_sigma']:.4f} sigma, "
              f"dchi2 {chk_a['delta_chi2']:+.3f}\nrendering check (B) margin: max {chk_b['max_dev_sigma']:.4f} sigma, "
              f"dchi2 {chk_b['delta_chi2']:+.3f}")
out = Path("outputs/m10d_fit_1A.png")
out.parent.mkdir(exist_ok=True)
plot_fit_summary(model, result, field=field, quasar_beta=qso.beta, zeropoint=img.zeropoint,
                 title="M10d: quasar 1A host, F115W (host Sersic + quasar + sky)", path=out,
                 errors=lap, extra_text=check_text)
print(f"\nsaved figure: {out}")
