"""Run a fit from a model YAML file and record it in the fit history (M13d).

    python -m forwardgal.run model.yaml [--starts N] [--max-iter N] [--no-errors] [--no-plot]

Loads the configuration (forwardgal.config), fits with its ``fit`` settings
(method, n_starts, seed, max_iter), computes Laplace errors, and writes a new
folder history_dir/NNNN_label (forwardgal.history) with model.yaml,
bestfit.yaml, result.yaml, arrays.npz, fit_summary.png and log.txt.
Long runs print their progress.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from forwardgal.config import absolute_paths, build, restore_radec
from forwardgal.fit import fit, laplace_errors
from forwardgal.history import FOLDER_RE, history_dir_for, new_fit_dir, save_fit
from forwardgal.serialize import read_yaml


def run(yaml_path, n_starts=None, max_iter=None, errors=True, plot=True, history_dir=None, progress=True, every=1,
        only=None):
    """Fit the model in ``yaml_path``; returns (folder, setup, result, laplace).

    only : None, or the names of the components to fit ("Fit selected", G4): all
        others are held at their values for this fit (edit.hold_others). The
        fit folder's model.yaml / bestfit.yaml keep the file's own free/locked
        settings; result.yaml lists the fitted components ("fitted_only").
    """
    yaml_path = Path(yaml_path).resolve()
    cfg = read_yaml(yaml_path)
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    t0 = time.perf_counter()
    setup = build(cfg, base_dir=yaml_path.parent)
    say(f"loaded {yaml_path.name}: {setup.model.params.size} free parameters (fluxes included), "
        f"{len(setup.model.names)} amplitudes, {int(setup.region.mask.sum())} fit pixels "
        f"({time.perf_counter() - t0:.0f} s)")
    if setup.fluxes_added:
        say(f"flux parameters added from one linear solve (F5): {', '.join(setup.fluxes_added)}")
    own = setup  # the file's own set-up (its free/locked settings are what the fit folder records)
    if only:
        from forwardgal.edit import hold_others

        setup = build(dict(own.cfg, model=hold_others(own.cfg["model"], only)), base_dir=yaml_path.parent)
        say(f"fitting only {', '.join(only)} ({setup.model.params.size} free parameters); the others are held")
    fs = dict(cfg.get("fit", {}))
    n = n_starts if n_starts is not None else fs.get("n_starts", 1)
    it = max_iter if max_iter is not None else fs.get("max_iter")
    for name, pt in setup.points.items():
        beta = f"; beta = ({pt.beta[0]:+.4f}, {pt.beta[1]:+.4f})\"" if setup.lensed else ""
        say(f"{name}: placed with {Path(pt.psf).name}, snapped by {pt.snap:.3f} px to its pixel grid{beta}")
    say(f"fitting: method {fs.get('method', 'lm')}, {n} start(s), seed {fs.get('seed', 0)}")
    result = fit(setup.model, n_starts=n, seed=fs.get("seed", 0), max_iter=it, method=fs.get("method", "lm"),
                 progress=progress, every=every)
    for k, (loss, its, conv) in enumerate(result.starts):
        say(f"  start {k}: loss {loss:.2f} after {its} iterations{'' if conv else ' (NOT converged)'}")
    lap = None
    if errors:
        lap = laplace_errors(setup.model, result)
        for w in lap.warnings:
            say(f"  Laplace warning: {w}")
    hd = Path(history_dir) if history_dir is not None else history_dir_for(yaml_path, cfg)
    folder = new_fit_dir(hd, cfg.get("label", yaml_path.stem))
    # The configuration AS RUN: flux parameters included (F5), RA/Dec points written as given.
    # (input paths absolute: the copy in the fit folder must find its files from there)
    as_run = absolute_paths(dict(own.cfg, model=restore_radec(own.cfg["model"], own.radec)), yaml_path.parent)
    # started from a history folder's model (e.g. a loaded bestfit.yaml saved there): record which (G3)
    started = yaml_path.parent.name if FOLDER_RE.match(yaml_path.parent.name) else None
    save_fit(folder, as_run, setup.model, result, setup.files, laplace=lap, zeropoint=setup.img.zeropoint,
             started_from=started, fitted_only=only)
    from forwardgal.export import imgblock  # data, model, residual, sigma, mask (GALFIT order)

    imgblock(setup, result, laplace=lap, fit_name=folder.name).writeto(folder / "imgblock.fits")
    if plot:
        from forwardgal.plotting import plot_fit_summary

        lensed = setup.lensed  # (no lens: no caustics, no source-plane panel content)
        plot_fit_summary(setup.model, result, field=setup.field if lensed else None,
                         quasar_beta=setup.quasar_beta if lensed else None, zeropoint=setup.img.zeropoint,
                         title=f"{folder.name}: {cfg.get('label', '')}", path=folder / "fit_summary.png",
                         errors=lap)
    say(f"best loss {result.loss:.2f}; saved {folder} ({time.perf_counter() - t0:.0f} s in total)")
    (folder / "log.txt").write_text("\n".join(log) + "\n")
    return folder, setup, result, lap


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit a forwardgal model YAML and record it in the fit history.")
    ap.add_argument("yaml")
    ap.add_argument("--starts", type=int, default=None, help="number of starts (default: fit.n_starts)")
    ap.add_argument("--max-iter", type=int, default=None, help="optimizer iterations (default: fit.max_iter)")
    ap.add_argument("--no-errors", action="store_true", help="skip the Laplace errors")
    ap.add_argument("--no-plot", action="store_true", help="skip the fit-summary figure")
    ap.add_argument("--quiet", action="store_true", help="do not print the parameters during the fit")
    ap.add_argument("--every", type=int, default=1, help="print the parameters every N iterations (default 1)")
    ap.add_argument("--only", default=None, help="fit only these components (comma-separated names); "
                                                 "the others are held at their values")
    a = ap.parse_args(argv)
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    run(a.yaml, n_starts=a.starts, max_iter=a.max_iter, errors=not a.no_errors, plot=not a.no_plot,
        progress=not a.quiet, every=a.every, only=only)


if __name__ == "__main__":
    main()
