"""Benchmark optimizers on the 1A bulge + disk model.

Methods (same 5 starts each; compiling done before timing):
  adam        : optax.adam, learning rate 0.01 in the unconstrained parameters
                (a common default, NOT tuned), up to 2000 iterations;
  adam+lbfgs  : 200 Adam iterations, then L-BFGS (up to 300);
  lbfgs       : optax L-BFGS with line search, up to 300 iterations;
  lm          : Levenberg-Marquardt (forwardgal default), up to 100 iterations.
For every run the loss is recorded after each iteration with its wall time.
Reported: time and iterations until loss <= best + 0.01 (the best loss any
run reached; delta chi^2 < 0.01), the final loss, and how many starts of each
method end in that same minimum.

Run from the repository root (long: ~20-30 min; behind a memory guard):
    python scripts/m10e_optimizer_benchmark.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import forwardgal  # noqa: F401
from common import PSF_FILE, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import _jittered_start, _lbfgs_step, lbfgs, levenberg_marquardt
from forwardgal.model import ForwardModel
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
bx, by = qso.beta
params = ModelParams([
    sersic_component("bulge", "source", bx + 0.005, by - 0.005, R_e=0.05, n=4.0, q=0.8, pa_deg=30.0),
    sersic_component("disk", "source", bx + 0.01, by - 0.01, R_e=0.25, n=1.0, q=0.5, pa_deg=30.0),
    sky_component(),
])
model = ForwardModel(params, grid, make_convolver(psf.kernel, shape), region.in_box(img.data),
                     region.in_box(img.sigma), region.in_box(region.mask), region.render_pixels_in_box(),
                     img.pixscale, fixed_templates={"QSO 1A": qso_template})

rng = np.random.default_rng(0)  # the same 5 starts as fit(n_starts=5, seed=0)
starts = [params.start()] + [_jittered_start(params, rng) for _ in range(4)]

# Adam: one compiled step.
adam = optax.adam(0.01)


@jax.jit
def adam_step(u, state):
    value, grad = model.value_and_grad(u)
    updates, state = adam.update(grad, state, u)
    return optax.apply_updates(u, updates), state, value


def run_adam(u, n_iter, trace, t0):
    state = adam.init(u)
    for it in range(1, n_iter + 1):
        u, state, value = adam_step(u, state)
        trace.append((time.perf_counter() - t0, it, float(value)))
    return u


# Warm-up (compiling), excluded from the timings.
print("compiling ...", flush=True)
tc = time.perf_counter()
run_adam(starts[0], 2, [], time.perf_counter())
lbfgs(model, starts[0], max_iter=2)
levenberg_marquardt(model, starts[0], max_iter=1)
print(f"compiled in {time.perf_counter() - tc:.0f} s", flush=True)

runs = {}
for method in ("lm", "lbfgs", "adam+lbfgs", "adam"):
    for k, u0 in enumerate(starts):
        trace = []
        t0 = time.perf_counter()
        cb = lambda it, loss: trace.append((time.perf_counter() - t0, it, loss))  # noqa: E731
        if method == "lm":
            levenberg_marquardt(model, u0, max_iter=100, callback=cb)
        elif method == "lbfgs":
            lbfgs(model, u0, max_iter=300, callback=cb)
        elif method == "adam":
            run_adam(u0, 2000, trace, t0)
        else:
            u = run_adam(u0, 200, trace, t0)
            offset = len(trace)
            lbfgs(model, u, max_iter=300, callback=lambda it, loss: trace.append((time.perf_counter() - t0, offset + it, loss)))
        runs[(method, k)] = trace
        print(f"{method:11s} start {k}: final loss {trace[-1][2]:10.2f} after {trace[-1][1]:5d} iterations, "
              f"{trace[-1][0]:7.1f} s", flush=True)

best = min(t[-1][2] for t in runs.values())
target = best + 0.01
lines = [f"best loss over all runs: {best:.3f}; target (delta chi2 < 0.01): {target:.3f}", "",
         f"{'method':11s} {'start':>5s} {'reached?':>9s} {'time [s]':>9s} {'iterations':>11s} {'final loss':>11s}"]
for (method, k), trace in runs.items():
    hit = next((t for t in trace if t[2] <= target), None)
    lines.append(f"{method:11s} {k:5d} {'yes' if hit else 'no':>9s} {hit[0] if hit else float('nan'):9.1f} "
                 f"{hit[1] if hit else -1:11d} {trace[-1][2]:11.3f}")
lines.append("")
for method in ("lm", "lbfgs", "adam+lbfgs", "adam"):
    finals = [runs[(method, k)][-1][2] for k in range(5)]
    same = sum(f <= target for f in finals)
    hits = [next((t for t in runs[(method, k)] if t[2] <= target), None) for k in range(5)]
    times = [h[0] for h in hits if h]
    lines.append(f"{method:11s}: {same}/5 starts end in the best minimum; "
                 f"median time to reach it {np.median(times) if times else float('nan'):.1f} s; "
                 f"total time for 5 starts {sum(runs[(method, k)][-1][0] for k in range(5)):.0f} s")
report = "\n".join(lines)
print("\n" + report)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/m10e_optimizer_benchmark.txt").write_text(report + "\n")
print("\nsaved: outputs/m10e_optimizer_benchmark.txt")
