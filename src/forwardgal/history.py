"""Fit history: one folder per fit (M13d).

Folders are named by a running number and a label (user, 2026-11-16), e.g.
``fits/0007_1A-bulge-disk``, in ``history_dir`` (default: ``fits/`` next to
the model YAML). Each holds:

    model.yaml    the exact configuration that was run (starting values)
    bestfit.yaml  the same configuration with the best-fit values as starting
                  values (to continue from; values exactly on a bound are
                  nudged inside it, and listed in result.yaml)
    result.yaml   readable summary: date, forwardgal version, loss, chi2/dof,
                  starts, best-fit values (incl. derived), amplitudes and
                  magnitudes, Laplace errors, input files with SHA-256
    arrays.npz    exact arrays: best-fit u, Laplace covariance and Hessian,
                  NUTS chains (if any)
    *.png, log    written by the run script

A DRAFT (G3, saved from the GUI without fitting) is a numbered folder in
the same sequence with model.yaml and draft.yaml (created, label,
started_from, chi2 / dof / n_pixels at saving, notes). ``list_history``
lists fits and drafts for the GUI table; ``trash`` moves a folder into
``history_dir/.trash/`` (recoverable with ``restore``; nothing is deleted).

BIC = chi2 + k ln N, with N the fit pixels and k = N - dof the fitted
quantities (shape parameters, flux parameters, solved amplitudes); chi2
stands for -2 ln L up to a constant. It needs n_pixels, recorded since G3.

Input data files are recorded by absolute path + SHA-256 (user's choice): on
reading a fit, moved or changed files give a warning.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import forwardgal
from forwardgal.serialize import read_yaml, write_yaml

FOLDER_RE = re.compile(r"^(\d{4})_(.+)$")


# --- input files ------------------------------------------------------------------


def sha256(path, chunk=2**24):
    """SHA-256 of a file (read in 16 MB chunks, so large mosaics are fine)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_record(files):
    """{role: Path} -> {role: {"path", "sha256", "bytes"}} (plain types)."""
    return {role: {"path": str(Path(p).resolve()), "sha256": sha256(p), "bytes": Path(p).stat().st_size}
            for role, p in files.items()}


def check_files(record):
    """Warnings (list of str) for recorded input files that are missing or changed."""
    out = []
    for role, r in record.items():
        p = Path(r["path"])
        if not p.exists():
            out.append(f"{role}: {p} no longer exists")
        elif p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            out.append(f"{role}: {p} has changed since the fit (SHA-256 differs)")
    return out


# --- folders ------------------------------------------------------------------------


def _safe_label(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-") or "fit"


def new_fit_dir(history_dir, label):
    """Create and return history_dir/NNNN_label with the next free number.

    Numbers are never reused: trashed folders (.trash/) count too, since other
    folders may name them (started_from).
    """
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    places = [history_dir] + ([history_dir / ".trash"] if (history_dir / ".trash").is_dir() else [])
    numbers = [int(m.group(1)) for d in places for p in d.iterdir() if (m := FOLDER_RE.match(p.name)) and p.is_dir()]
    folder = history_dir / f"{(max(numbers) + 1 if numbers else 1):04d}_{_safe_label(label)}"
    folder.mkdir()
    return folder


# --- writing ----------------------------------------------------------------------


def _plain(x):
    """numpy / JAX scalars and arrays -> Python floats and lists."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def bestfit_config(cfg, result):
    """Copy of cfg with each component parameter's value set to its best fit.

    Returns (cfg_new, nudged): values exactly on a bound (e.g. n = 0.5 when the
    fit sits at the bound) are moved inside by 1e-9 x max(1, |bound|), since a
    starting value must lie strictly inside; ``nudged`` lists them.
    """
    import copy

    new = copy.deepcopy(cfg)
    nudged = []
    for comp in new["model"]["components"]:
        vals = result.values.get(comp["name"], {})
        for key, pd in comp["params"].items():
            if key not in vals:
                continue
            x = float(vals[key])
            lo, hi = pd.get("lo"), pd.get("hi")
            if lo is not None and x <= lo:
                x = lo + 1e-9 * max(1.0, abs(lo))
                nudged.append(f"{comp['name']}.{key}")
            if hi is not None and x >= hi:
                x = hi - 1e-9 * max(1.0, abs(hi))
                nudged.append(f"{comp['name']}.{key}")
            pd["value"] = x
    return new, nudged


def save_fit(folder, cfg, model, result, files, laplace=None, posterior=None, zeropoint=None, notes=None,
             started_from=None, fitted_only=None):
    """Write model.yaml, bestfit.yaml, result.yaml and arrays.npz into ``folder``.

    cfg : the configuration dictionary that was run (forwardgal.config)
    model, result : the ForwardModel and its FitResult (forwardgal.fit)
    files : {role: path} of the input files (Setup.files)
    laplace : LaplaceErrors or None; posterior : Posterior (NUTS) or None
    zeropoint : for magnitudes in result.yaml (default: none written)
    started_from : name of the history folder the run started from, or None
    fitted_only : the components fitted ("Fit selected"; the others held), or None (all)
    """
    folder = Path(folder)
    write_yaml(cfg, folder / "model.yaml", header=f"forwardgal model, as run ({folder.name})")
    best_cfg, nudged = bestfit_config(cfg, result)
    write_yaml(best_cfg, folder / "bestfit.yaml", header=f"forwardgal model with the best-fit values of {folder.name}")

    c2, dof = model.chi2_dof(jnp.asarray(result.u))
    amps = {k: float(v) for k, v in result.amplitudes.items()}
    summary = {
        "fit": folder.name,
        "label": cfg.get("label", ""),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "forwardgal_version": forwardgal.__version__,
        "loss": float(result.loss),
        "chi2": float(c2), "dof": int(dof), "chi2_per_dof": float(c2 / dof),
        "n_pixels": int(np.asarray(model.mask).sum()),
        "seconds": float(result.seconds),
        "starts": [{"loss": float(loss), "iterations": int(it), "converged": bool(conv)}
                   for loss, it, conv in result.starts],
        "values": _plain(result.values),
        "amplitudes": amps,
    }
    if zeropoint is not None:
        summary["magnitudes"] = {k: float(-2.5 * np.log10(a) + zeropoint) for k, a in amps.items() if a > 0
                                 and k != "sky"}
        summary["zeropoint"] = float(zeropoint)
    if nudged:
        summary["bestfit_nudged_inside_bounds"] = nudged
    if laplace is not None:
        summary["laplace"] = {
            "scale": float(laplace.scale),
            "parameters": {n: {"best": float(b), "sigma": float(s)}
                           for n, b, s in zip(laplace.names, laplace.best, laplace.sigma)},
            "derived": {n: {"best": float(b), "sigma": float(s)} for n, b, s in (laplace.derived or [])},
            "warnings": list(laplace.warnings),
        }
    if posterior is not None:
        flat = posterior.samples
        p16, p50, p84 = np.percentile(flat, [15.865, 50.0, 84.135], axis=0)
        summary["nuts"] = {"chains": int(posterior.chains.shape[0]), "samples": int(posterior.chains.shape[1]),
                           "divergences": int(posterior.divergences), "seconds": float(posterior.seconds),
                           # (R-hat compares chains: undefined, None, for one chain)
                           "max_r_hat": float(np.nanmax(posterior.r_hat)) if np.isfinite(posterior.r_hat).any()
                           else None,
                           # the posterior: median and the 68 % interval (16th, 84th percentiles) per parameter
                           "parameters": {n: {"median": float(m), "p16": float(lo), "p84": float(hi),
                                              "r_hat": float(r), "n_eff": float(e)}
                                          for n, m, lo, hi, r, e in zip(posterior.names, p50, p16, p84,
                                                                        posterior.r_hat, posterior.n_eff)}}
    summary["input_files"] = file_record(files)
    if notes:
        summary["notes"] = str(notes)
    if started_from:
        summary["started_from"] = str(started_from)
    if fitted_only:
        summary["fitted_only"] = [str(n) for n in fitted_only]
    write_yaml(summary, folder / "result.yaml", header=f"forwardgal fit result ({folder.name})")

    arrays = {"u": np.asarray(result.u), "free_keys": np.array(model.params.free_keys)}
    if laplace is not None:
        arrays.update(laplace_names=np.array(laplace.names), covariance=laplace.covariance)
        if laplace.hessian is not None:
            arrays["hessian"] = laplace.hessian
    if posterior is not None:
        arrays.update(posterior_names=np.array(posterior.names), posterior_chains=posterior.chains)
    np.savez(folder / "arrays.npz", **arrays)
    return folder


# --- reading ----------------------------------------------------------------------


def read_fit(folder, check=True):
    """Read a fit folder -> dict(config, bestfit, result, arrays, warnings).

    With check=True the recorded input files are compared with their
    SHA-256; differences are returned in "warnings" and issued as warnings.
    """
    folder = Path(folder)
    out = {"folder": folder, "config": read_yaml(folder / "model.yaml"),
           "bestfit": read_yaml(folder / "bestfit.yaml"), "result": read_yaml(folder / "result.yaml")}
    with np.load(folder / "arrays.npz") as z:
        out["arrays"] = {k: z[k] for k in z.files}
    out["warnings"] = check_files(out["result"].get("input_files", {})) if check else []
    for w in out["warnings"]:
        warnings.warn(w, stacklevel=2)
    return out


def list_fits(history_dir):
    """One summary dict per fit folder, in number order (for the GUI history)."""
    rows = []
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return rows
    for p in sorted(history_dir.iterdir()):
        m = FOLDER_RE.match(p.name)
        if not (m and p.is_dir() and (p / "result.yaml").exists()):
            continue
        r = read_yaml(p / "result.yaml")
        rows.append({"number": int(m.group(1)), "folder": p.name, "label": r.get("label", ""),
                     "created": r.get("created", ""), "loss": r.get("loss"), "chi2_per_dof": r.get("chi2_per_dof"),
                     "converged": all(s["converged"] for s in r.get("starts", [])) if r.get("starts") else None})
    return rows


def history_dir_for(yaml_path, cfg):
    """The history folder of a model file: cfg["history_dir"] (relative to the
    YAML's folder) or ``fits/`` next to the YAML (user's choice). A YAML inside
    a history folder (fits/NNNN_label/model.yaml, bestfit.yaml) belongs to
    that same history (G3)."""
    base = Path(yaml_path).parent
    hd = cfg.get("history_dir")
    if hd:
        return (base / Path(hd).expanduser()).resolve()
    if FOLDER_RE.match(base.name) and ((base / "result.yaml").exists() or (base / "draft.yaml").exists()):
        return base.parent
    return base / "fits"


# --- drafts, the history table, trash (G3) ---------------------------------------------


def save_draft(history_dir, cfg, label=None, started_from=None, chi2=None, dof=None, n_pixels=None, notes=None):
    """Save an unfitted model (the GUI's "Save to history") as a numbered draft folder; returns it.

    cfg : the whole configuration (its model: section is what was built in the GUI)
    started_from : name of the history folder the GUI had loaded (or None)
    chi2, dof, n_pixels : the live model's at saving (for the table), or None
    """
    label = label or cfg.get("label", "draft")
    folder = new_fit_dir(history_dir, label)
    write_yaml(cfg, folder / "model.yaml", header=f"forwardgal model, draft ({folder.name}, not fitted)")
    info = {"draft": folder.name, "label": str(label),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "forwardgal_version": forwardgal.__version__}
    if started_from:
        info["started_from"] = str(started_from)
    if chi2 is not None:
        info.update(chi2=float(chi2), dof=int(dof), chi2_per_dof=float(chi2) / int(dof))
    if n_pixels is not None:
        info["n_pixels"] = int(n_pixels)
    if notes:
        info["notes"] = str(notes)
    write_yaml(info, folder / "draft.yaml", header="forwardgal draft: a model saved from the GUI, not fitted")
    return folder


def components_text(model_spec):
    """Short description of a model spec, e.g. "2 Sérsic + point + sky" ("+ Fourier", "+ bending")."""
    comps = model_spec.get("components", [])
    n = {k: sum(c["kind"] == k for c in comps) for k in ("sersic", "point", "sky")}
    parts = []
    for kind, word in (("sersic", "Sérsic"), ("point", "point"), ("sky", "sky")):
        if n[kind]:
            parts.append(word if n[kind] == 1 and kind != "sersic" else f"{n[kind]} {word}")
    keys = {k for c in comps for k in c.get("params", {})}
    if any(re.fullmatch(r"[cs]\d|a\d|phi\d", k) for k in keys):
        parts.append("Fourier")
    if any(re.fullmatch(r"b\d", k) for k in keys):
        parts.append("bending")
    return " + ".join(parts) if parts else "empty"


def bic(chi2, dof, n_pixels):
    """chi2 + k ln N with k = N - dof (see the module docstring); None if anything is missing."""
    if chi2 is None or dof is None or not n_pixels:
        return None
    return float(chi2) + (int(n_pixels) - int(dof)) * float(np.log(int(n_pixels)))


def list_history(history_dir):
    """One row per fit or draft folder of ``history_dir``, in number order (the GUI's history table).

    Row keys: number, folder (name), path, kind ("fit" / "draft"), label, created,
    components (components_text), started_from, status ("unfitted", "optimizer",
    "optimizer (not converged)", "NUTS"), chi2_per_dof, bic (None when unknown).
    """
    rows = []
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return rows
    for p in sorted(history_dir.iterdir()):
        m = FOLDER_RE.match(p.name)
        if not (m and p.is_dir() and (p / "model.yaml").exists()):
            continue
        if (p / "result.yaml").exists():
            info, kind = read_yaml(p / "result.yaml"), "fit"
            starts = info.get("starts") or []
            status = "NUTS" if "nuts" in info else (
                "optimizer" if not starts or any(s["converged"] for s in starts) else "optimizer (not converged)")
            if info.get("fitted_only"):
                status += f" · only {', '.join(info['fitted_only'])}"
        elif (p / "draft.yaml").exists():
            info, kind, status = read_yaml(p / "draft.yaml"), "draft", "unfitted"
        else:
            continue
        from forwardgal.config import upgrade_config  # (an old quasar: section counts as a point)

        cfg = upgrade_config(read_yaml(p / "model.yaml"))
        rows.append({"number": int(m.group(1)), "folder": p.name, "path": p, "kind": kind,
                     "label": info.get("label", ""), "created": info.get("created", ""),
                     "components": components_text(cfg.get("model", {})),
                     "started_from": info.get("started_from"), "status": status,
                     "chi2_per_dof": info.get("chi2_per_dof"),
                     "bic": bic(info.get("chi2"), info.get("dof"), info.get("n_pixels"))})
    return rows


TRASH = ".trash"


def trash(history_dir, folder):
    """Move fit or draft ``folder`` (its name) into history_dir/.trash/ (nothing is deleted); returns the new path."""
    src = Path(history_dir) / folder
    if not (src.is_dir() and FOLDER_RE.match(src.name)):
        raise FileNotFoundError(f"no fit or draft folder {folder!r} in {history_dir}")
    bin_ = Path(history_dir) / TRASH
    bin_.mkdir(exist_ok=True)
    dst = bin_ / src.name
    k = 1
    while dst.exists():  # the same name trashed before: keep both
        dst = bin_ / f"{src.name}~{k}"
        k += 1
    src.rename(dst)
    return dst


def list_trash(history_dir):
    """Names of the trashed folders (as in .trash/), sorted."""
    bin_ = Path(history_dir) / TRASH
    return sorted(p.name for p in bin_.iterdir() if p.is_dir()) if bin_.exists() else []


def restore(history_dir, name):
    """Move a trashed folder back (under its original name; refused if that name is taken again)."""
    src = Path(history_dir) / TRASH / name
    if not src.is_dir():
        raise FileNotFoundError(f"nothing called {name!r} in the trash")
    dst = Path(history_dir) / name.split("~")[0]
    if dst.exists():
        raise FileExistsError(f"{dst.name} exists again; rename or trash it first")
    src.rename(dst)
    return dst
