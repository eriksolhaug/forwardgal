"""Milestone 12f: a BENT arm component for the arm south of quasar 1A (F115W).

The bulge + disk model (scripts/m10d_fit_1A_bulge_disk.py) leaves a curved
positive residual south of the quasar (theta ~ (17.8", 2.6")). Traced to the
source plane (user, 2026-11-16: place the component there): centre beta ~
(-5.836, -0.119), length ~0.34", PA ~171 deg, sagitta ~0.018". Three fits,
all started at the M11c NUTS medians for the bulge and disk:
  (A) bulge + disk (reference, 1 LM start);
  (B) + ARM: a thin Sersic at the traced position, unbent (3 starts);
  (C) + ARM with bending modes b2, b3 (3 starts).
Reports delta chi^2 (A-B: the arm; B-C: its bending), Laplace errors and a
rendering check. Writes outputs/m12f_fit_1A_bending.png (fit summary of C)
and outputs/m12f_residuals_1A.png (residuals of A, B, C).

Run from the repository root (a few minutes, ~2.5 GB; behind a memory guard):
    python scripts/m12f_fit_1A_bending.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import forwardgal  # noqa: F401
from common import PSF_FILE, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.components import ModelParams, bending_of, sersic_component, sky_component
from forwardgal.diagnostics import rendering_check
from forwardgal.fit import fit, laplace_errors
from forwardgal.model import ForwardModel
from forwardgal.params import Param
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

# Start: the NUTS medians of M11c (deeper minimum than LM's own, loss 10063.5).
N_START = {"n": Param(0.55, lo=0.5, hi=8.0)}  # disk n sat at the 0.5 bound


ARM_START = dict(x0=-5.836, y0=-0.119, R_e=0.17, n=1.0, q=0.25, pa_deg=171.0)  # traced from the residual


def build(arm=None):
    """arm: None (bulge + disk), "plain" or "bent" (b2, b3)."""
    bulge = sersic_component("bulge", "source", -5.7265, 0.21959, R_e=0.0494, n=0.906, q=0.863, pa_deg=20.1)
    disk = sersic_component("disk", "source", -5.7594, 0.13714, R_e=0.393, n=0.55, q=0.752, pa_deg=16.0,
                            overrides=N_START)
    comps = [bulge, disk]
    if arm is not None:
        comps.append(sersic_component("arm", "source", **ARM_START, bending=[2, 3] if arm == "bent" else None))
    params = ModelParams(comps + [sky_component()])
    return ForwardModel(params, grid, convolve, data, sigma, mask, pixels, img.pixscale,
                        fixed_templates={"QSO 1A": qso_template})


fits = {}
for label, arm, starts in [("A", None, 1), ("B", "plain", 3), ("C", "bent", 3)]:
    t0 = time.perf_counter()
    model = build(arm)
    res = fit(model, n_starts=starts, seed=0)
    fits[label] = (model, res)
    print(f"({label}) arm = {arm}: loss {res.loss:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    for k, (loss, it, conv) in enumerate(res.starts):
        print(f"     start {k}: loss {loss:.2f}, {it} iterations{'' if conv else ' (NOT converged)'}", flush=True)
model_b, res_b = fits["C"]
print("\n" + res_b.describe(model_b), flush=True)
d_arm = fits["A"][1].loss - fits["B"][1].loss
d_bend = fits["B"][1].loss - fits["C"][1].loss
print(f"\ndelta chi^2: arm (A - B) = {d_arm:.1f} for 7 extra parameters + 1 flux; "
      f"bending (B - C) = {d_bend:.1f} for 2 extra parameters", flush=True)
lap = laplace_errors(model_b, res_b)
print("\n" + lap.describe(zeropoint=img.zeropoint), flush=True)

# Rendering check of the bent host (hybrid vs exact on the render pixels).
hybrid, exact = np.zeros(shape), np.zeros(shape)
for comp in model_b.params.components[:3]:
    v = {k: float(x) for k, x in res_b.values[comp.name].items()}
    ie = res_b.amplitudes[comp.name] * float(unit_flux_ie(v["R_e"], v["n"], v["q"], img.pixscale))
    args = (v["x0"], v["y0"], ie, v["R_e"], v["n"], v["q"], v["pa_deg"])
    b = bending_of(v, comp)
    hybrid += np.asarray(render_lensed_sersic(grid, *args, pixels=pixels, bending=b))
    exact += np.asarray(render_lensed_sersic_exact(grid, *args, bending=b, bend_split=8))  # batched: ~1.2 GB
rmask = np.zeros(shape, bool)
rmask[pixels] = True
chk = rendering_check(hybrid, np.where(rmask, exact, 0.0), data, sigma, mask, convolve=convolve)
check = f"rendering check, hybrid vs exact (bend_split 8): max {chk['max_dev_sigma']:.4f} sigma, dchi2 {chk['delta_chi2']:+.3f}"
print(check)

Path("outputs").mkdir(exist_ok=True)
plot_fit_summary(model_b, res_b, field=field, quasar_beta=qso.beta, zeropoint=img.zeropoint,
                 title="M12f: quasar 1A host, bulge + disk + BENT ARM (b2, b3) + quasar + sky", errors=lap,
                 path="outputs/m12f_fit_1A_bending.png",
                 extra_text=f"delta chi2: arm {d_arm:.1f}, its bending {d_bend:.1f}\n" + check)

import jax.numpy as jnp  # noqa: E402

tx, ty = model_b.grid.theta_corner_x, model_b.grid.theta_corner_y
ext = (tx[0, 0], tx[0, -1], ty[0, 0], ty[-1, 0])
fig, ax = plt.subplots(1, 3, figsize=(14, 6))
for a, (lab, key) in zip(ax, [("(A) bulge + disk", "A"), ("(B) + arm", "B"), ("(C) + bent arm", "C")]):
    m, r = fits[key]
    resid = np.where(mask, (data - np.asarray(m.model_image(jnp.asarray(r.u)))) / sigma, np.nan)
    im = a.imshow(resid, origin="lower", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
    a.set_title(f"{lab}: chi2 = {r.loss:.1f}", fontsize=10)
    a.set_xlabel("theta_x [arcsec, +east]")
    a.set_ylabel("theta_y [arcsec, +north]")
fig.colorbar(im, ax=ax, fraction=0.03, label="(data - model) / sigma")
fig.savefig("outputs/m12f_residuals_1A.png", dpi=100, bbox_inches="tight")
print("saved outputs/m12f_fit_1A_bending.png, outputs/m12f_residuals_1A.png")
