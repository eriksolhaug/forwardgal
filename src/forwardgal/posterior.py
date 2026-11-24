"""NUTS posteriors of a fit in the history (the GUI's "Run NUTS -> posteriors", 2026-11-23).

    python -m forwardgal.posterior fits/0007_1A-bulge-disk --warmup 500 --samples 1000 --chains 2

The fit folder's model.yaml (the configuration as run) is built again, and
NUTS (forwardgal.sample.nuts) starts at the fit's exact best point (u in
arrays.npz; the starting mass matrix is the Laplace curvature there). A fit
of selected components ("fitted_only" in result.yaml) is sampled the same
way: the others held at their values. A NEW history folder is written,
started_from = the fit: the same best point (bestfit.yaml), Laplace errors,
the posterior (result.yaml "nuts": medians and 16 / 84 % percentiles;
arrays.npz: the chains), corner.png (with the Laplace 1-sigma ellipses) and
imgblock.fits.

NUTS explores the posterior around the minimum it starts in (M11c): if the
fit sits in a false minimum, so do the samples (the M17 safeguard, later).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from forwardgal.config import build, restore_radec
from forwardgal.fit import FitResult, _with_derived, laplace_errors
from forwardgal.history import new_fit_dir, save_fit
from forwardgal.serialize import read_yaml


def result_at(model, u, starts=(), seconds=0.0):
    """FitResult of ``model`` at the parameter vector ``u`` (e.g. a saved fit's best point)."""
    import jax.numpy as jnp

    u = jnp.asarray(u)
    _, a = model.solve(u)
    return FitResult(u=np.asarray(u), loss=float(model.loss(u)),
                     values=_with_derived(model.params, model.params.unpack(u)),
                     amplitudes={n: float(x) for n, x in zip(model.names, np.asarray(a))},
                     starts=list(starts), seconds=seconds)


def run_nuts(fit_folder, warmup=500, samples=1000, chains=2, seed=0, progress=True, corner=True):
    """NUTS from the fit in ``fit_folder``; writes and returns the new history folder (and the Posterior)."""
    from forwardgal.export import imgblock
    from forwardgal.sample import nuts

    fit_folder = Path(fit_folder).resolve()
    cfg = read_yaml(fit_folder / "model.yaml")
    info = read_yaml(fit_folder / "result.yaml")
    only = info.get("fitted_only")
    t0 = time.perf_counter()
    own = build(cfg, base_dir=fit_folder)
    setup = own
    if only:
        from forwardgal.edit import hold_others

        setup = build(dict(own.cfg, model=hold_others(own.cfg["model"], only)), base_dir=fit_folder)
    model = setup.model
    with np.load(fit_folder / "arrays.npz") as z:
        u, keys = z["u"], [str(k) for k in z["free_keys"]]
    if keys != list(model.params.free_keys):
        raise ValueError(f"{fit_folder.name}: its saved parameters {keys} do not match the model's "
                         f"{list(model.params.free_keys)}")
    result = result_at(model, u, starts=[(s["loss"], s["iterations"], s["converged"]) for s in info.get("starts", [])])
    print(f"{fit_folder.name}: loss {result.loss:.2f} at the fit's best point ({model.params.size} free "
          f"parameters{'; only ' + ', '.join(only) if only else ''}); Laplace errors ...", flush=True)
    lap = laplace_errors(model, result)
    print(f"NUTS: {chains} chain(s) x ({warmup} warm-up + {samples} samples) ...", flush=True)
    post = nuts(model, result, num_warmup=warmup, num_samples=samples, num_chains=chains, seed=seed,
                progress=progress)
    print(post.describe(zeropoint=setup.img.zeropoint), flush=True)

    folder = new_fit_dir(fit_folder.parent, cfg.get("label", fit_folder.name) + "-nuts")
    as_run = dict(own.cfg, model=restore_radec(own.cfg["model"], own.radec))
    save_fit(folder, as_run, model, result, own.files, laplace=lap, posterior=post, zeropoint=setup.img.zeropoint,
             started_from=fit_folder.name, fitted_only=only,
             notes=f"NUTS {chains} x ({warmup} + {samples}), seed {seed}, from {fit_folder.name}")
    imgblock(setup, result, laplace=lap, fit_name=folder.name).writeto(folder / "imgblock.fits")
    if corner:  # the sampled parameters (held components' fluxes are constant: nothing to show)
        keys = [n for n in post.names if np.ptp(post.column(n)) > 0]
        post.corner(keys=keys, laplace=lap, path=folder / "corner.png")
    print(f"NUTS done in {time.perf_counter() - t0:.0f} s, {post.divergences} divergences; saved {folder}", flush=True)
    return folder, post


def main(argv=None):
    ap = argparse.ArgumentParser(description="NUTS posteriors of a fit in the history (a new history folder).")
    ap.add_argument("fit_folder")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", action="store_true", help="no progress bar")
    ap.add_argument("--no-corner", action="store_true", help="no corner plot")
    a = ap.parse_args(argv)
    run_nuts(a.fit_folder, a.warmup, a.samples, a.chains, a.seed, progress=not a.no_progress,
             corner=not a.no_corner)


if __name__ == "__main__":
    main()
