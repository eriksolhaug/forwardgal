"""Milestone 10c: recover a known host from synthetic data on the REAL setup.

Real traced grid, example mask (+35 px render margin), PSF, ERR noise map and
the image-fixed quasar 1A (PSF placed with no shift). Synthetic data = known
host (lensed Sersic) + quasar + sky, + Gaussian noise drawn from the ERR map.
The fit starts from perturbed values (3 starts) and must recover the truth.

Writes outputs/m10c_synthetic_fit.png (data/mask, model, residuals, source
plane, numbers).

Run from the repository root (heavy: run behind a memory guard if unsure):
    python scripts/m10c_synthetic_real.py
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import forwardgal  # noqa: F401
from common import PSF_FILE, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import fit
from forwardgal.model import ForwardModel
from forwardgal.plotting import plot_fit_summary
from forwardgal.psf import load_psf, make_convolver, make_point_source_renderer
from forwardgal.quasar import image_fixed_quasar, snap_to_psf
from forwardgal.regions import fit_region
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
sigma = region.in_box(img.sigma)
pixels = region.render_pixels_in_box()

# Truth: a plausible host near the quasar (fluxes as found in M8).
TRUTH = dict(x0=qso.beta[0] + 0.02, y0=qso.beta[1] - 0.015, R_e=0.12, n=1.5, q=0.6, pa_deg=30.0)
TRUE_FLUX = {"host": 65.0, "sky": 0.042, "QSO 1A": 237.0}


def build(host, data):
    params = ModelParams([sersic_component("host", "source", **host), sky_component()])
    return ForwardModel(params, grid, convolve, data, sigma, mask, pixels, img.pixscale,
                        fixed_templates={"QSO 1A": qso_template})


truth_model = build(TRUTH, np.zeros(shape))
T = truth_model.templates(truth_model.params.start())
clean = np.asarray(jnp.tensordot(jnp.array([TRUE_FLUX[n] for n in truth_model.names]), T, axes=1))
data = clean + np.random.default_rng(42).normal(0.0, sigma)

start = dict(x0=TRUTH["x0"] - 0.02, y0=TRUTH["y0"] + 0.015, R_e=0.18, n=2.5, q=0.75, pa_deg=55.0)
model = build(start, data)
result = fit(model, n_starts=3, seed=0)
print(result.describe(model))

print(f"\n{'parameter':10s} {'truth':>10s} {'fit':>10s}")
for k, t in TRUTH.items():
    print(f"{k:10s} {t:10.4f} {float(result.values['host'][k]):10.4f}")
for n, t in TRUE_FLUX.items():
    print(f"{'flux ' + n:10s} {t:10.4f} {result.amplitudes[n]:10.4f}")

out = Path("outputs/m10c_synthetic_fit.png")
out.parent.mkdir(exist_ok=True)
plot_fit_summary(model, result, field=field, quasar_beta=qso.beta, zeropoint=img.zeropoint,
                 title="M10c: synthetic host + quasar 1A + sky on the real setup", path=out)
print(f"\nsaved figure: {out}")
