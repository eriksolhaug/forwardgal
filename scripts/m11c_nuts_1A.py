"""Milestone 11c: NUTS posterior of the 1A bulge + disk model (F115W).

Same model as scripts/m10d_fit_1A_bulge_disk.py. LM best fit (5 starts),
Laplace errors, then NUTS: 2 chains x (500 warm-up + 1000 samples), Laplace
mass matrix. NOTE: NUTS explores the posterior around the minimum it starts
in; LM's best minimum here (loss ~10139) is NOT the deepest one found in the
optimizer benchmark (~10063.5, Adam -> L-BFGS; deferred until after M15).

Writes outputs/m11c_nuts_1A.txt (summary, NUTS vs Laplace) and
outputs/m11c_corner_1A.png. Progress bar on stderr.

Run from the repository root (long: ~40-50 min; behind a memory guard):
    python scripts/m11c_nuts_1A.py
"""

from pathlib import Path

import numpy as np

import forwardgal  # noqa: F401
from common import PSF_FILE, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import fit, laplace_errors
from forwardgal.model import ForwardModel
from forwardgal.psf import load_psf, make_convolver, make_point_source_renderer
from forwardgal.quasar import image_fixed_quasar, snap_to_psf
from forwardgal.regions import fit_region
from forwardgal.sample import nuts
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
bx, by = qso.beta
params = ModelParams([
    sersic_component("bulge", "source", bx + 0.005, by - 0.005, R_e=0.05, n=4.0, q=0.8, pa_deg=30.0),
    sersic_component("disk", "source", bx + 0.01, by - 0.01, R_e=0.25, n=1.0, q=0.5, pa_deg=30.0),
    sky_component(),
])
model = ForwardModel(params, grid, make_convolver(psf.kernel, shape), region.in_box(img.data),
                     region.in_box(img.sigma), region.in_box(region.mask), region.render_pixels_in_box(),
                     img.pixscale, fixed_templates={"QSO 1A": qso_template})

result = fit(model, n_starts=5, seed=0)
print(result.describe(model), flush=True)
lap = laplace_errors(model, result)
print("\n" + lap.describe(zeropoint=img.zeropoint), flush=True)

post = nuts(model, result, num_warmup=500, num_samples=1000, num_chains=2, seed=0, progress=True)
text = [post.describe(zeropoint=img.zeropoint), "",
        "NOTE: posterior around LM's best minimum (loss %.2f); a deeper minimum (~10063.5) exists "
        "(optimizer benchmark)." % result.loss, "",
        f"{'parameter':28s} {'NUTS sigma':>11s} {'Laplace sigma':>14s} {'ratio':>6s}"]
for name in post.names:
    if name in lap.names:
        s_n = np.std(post.column(name))
        s_l = lap.sigma[lap.names.index(name)]
        text.append(f"{name:28s} {s_n:11.4g} {s_l:14.4g} {s_n / s_l if s_l > 0 else np.nan:6.2f}")
report = "\n".join(text)
print("\n" + report, flush=True)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/m11c_nuts_1A.txt").write_text(report + "\n")
keys = ["bulge.R_e", "bulge.n", "bulge.q", "disk.R_e", "disk.n", "disk.q", "amp:bulge", "amp:disk", "amp:QSO 1A", "amp:sky"]
post.corner(keys=[k for k in keys if k in post.names], laplace=lap, path="outputs/m11c_corner_1A.png")
print("saved: outputs/m11c_nuts_1A.txt, outputs/m11c_corner_1A.png", flush=True)
