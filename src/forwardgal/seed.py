"""Seed a fit from another fit's best values, with limited movement (M14).

User's design (2026-11-16): no joint multi-band optimization. A model for
band B starts from the best fit of band A, and each parameter may move only
within a user-defined range of band A's best value. Fluxes are solved per
band as usual (they are not parameters here).

Ranges, per component name ("*" = default for all components; a component's
own entry overrides "*" key by key)::

    {"*":    {"position": 0.02, "R_e": {"rel": 0.2}, "n": 0.5, "q": 0.05, "pa_deg": 10},
     "disk": {"n": 1.0}}

  number          absolute half-width: bounds (best - w, best + w)
  {"rel": f}      relative: w = f |best|
  "position": r   the centre (x0, y0) stays within r arcsec of the best
                  centre, in the component's own plane (a circle; "within")
  "q", "pa_deg"   for an (e1, e2) Sersic: the component switches to (q, PA)
                  with bounds q +- dq (inside 0.05..1) and PA +- dPA (a PA
                  without a range gets +-90 deg: no 0/180 seam)

Bounds from a range are intersected with the parameter's own bounds (e.g.
0.5 < n < 8). Parameters WITHOUT a range start at the best value and stay
free within their usual bounds (user's choice). Values exactly on a bound are
nudged inside. Fixed parameters stay fixed (at the best value).
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from forwardgal.components import (
    FLUX_KEYS,
    FOURIER_SUM_MAX,
    Q_MIN,
    FourierBound,
    ModelParams,
    PositionConstraint,
    within,
)
from forwardgal.params import Param
from forwardgal.serialize import model_from_dict, model_to_dict, read_yaml



def _width(spec, best):
    if isinstance(spec, dict):
        if set(spec) != {"rel"}:
            raise ValueError(f"range spec must be a number or {{'rel': f}}, got {spec!r}")
        return float(spec["rel"]) * abs(best)
    return float(spec)


def _inside(x, lo, hi):
    """x moved inside the open interval (lo, hi) if it sits on (or beyond) a bound."""
    eps = 1e-9
    if lo is not None and x <= lo:
        x = lo + eps * max(1.0, abs(lo))
    if hi is not None and x >= hi:
        x = hi - eps * max(1.0, abs(hi))
    return x


def _bounded(prm, best, spec):
    """Param starting at best, with bounds (best -+ w) intersected with prm's own."""
    lo, hi = prm.lo, prm.hi
    if spec is not None:
        w = _width(spec, best)
        if w <= 0:
            raise ValueError(f"range width must be positive, got {w}")
        lo = best - w if lo is None else max(lo, best - w)
        hi = best + w if hi is None else min(hi, best + w)
    return Param(_inside(float(best), lo, hi), free=prm.free, lo=lo, hi=hi, prior_mu=prm.prior_mu,
                 prior_sigma=prm.prior_sigma)


def seed_params(base_model, best_values, ranges=None):
    """Band B's starting model from band A's model and best-fit values.

    base_model : ModelParams or model dict (band A's model.yaml "model")
    best_values : {component: {param: value}} incl. derived q, pa_deg (a
        FitResult.values or result.yaml "values")
    ranges : see the module docstring (None: no ranges, all free)
    Returns a ModelParams.
    """
    base = model_from_dict(base_model) if isinstance(base_model, dict) else base_model
    ranges = dict(ranges or {})
    unknown = set(ranges) - {"*"} - set(base.by_name)
    if unknown:
        raise ValueError(f"ranges for unknown component(s) {sorted(unknown)}")
    comps, extra = [], []
    for c in base.components:
        best = {k: float(v) for k, v in best_values[c.name].items()}
        applicable = set(c.params) | ({"position"} if "x0" in c.params else set())
        if c.kind == "sersic":
            applicable |= {"q", "pa_deg"}
        own = ranges.get(c.name, {})
        bad = set(own) - applicable  # in the component's own entry: a typo -> error
        if bad:
            raise ValueError(f"{c.name}: range for unknown parameter(s) {sorted(bad)}")
        # "*" keys that do not apply to this component (e.g. R_e for the sky) are ignored
        spec = {k: v for k, v in {**ranges.get("*", {}), **own}.items() if k in applicable}
        to_qpa = c.kind == "sersic" and "e1" in c.params and ("q" in spec or "pa_deg" in spec)
        params = {}
        for k, prm in c.params.items():
            if to_qpa and k in ("e1", "e2"):
                continue
            if not prm.free:
                params[k] = Param(best[k], free=False)
                continue
            params[k] = _bounded(prm, best[k], spec.get(k))
        if to_qpa:
            q, pa = best["q"], best["pa_deg"]
            dq, dpa = spec.get("q"), spec.get("pa_deg")
            lo_q = Q_MIN if dq is None else max(Q_MIN, q - _width(dq, q))
            hi_q = 1.0 if dq is None else min(1.0, q + _width(dq, q))
            params["q"] = Param(_inside(q, lo_q, hi_q), lo=lo_q, hi=hi_q)
            w = 90.0 if dpa is None else min(90.0, _width(dpa, pa))
            params["pa_deg"] = Param(_inside(pa, pa - w, pa + w), lo=pa - w, hi=pa + w)
        # Joint constraints that came with the component: keep, except the |e|
        # bound when switching to (q, PA), and the Fourier bound when a (c, s)
        # range bounds the modes (it needs them unbounded; check it is safe).
        cons = []
        for x in c.constraints:
            if isinstance(x, PositionConstraint) and x.keys == ("e1", "e2") and to_qpa:
                continue
            if isinstance(x, FourierBound) and any(params[k].lo is not None or params[k].hi is not None
                                                   for k in x.keys):
                worst = sum(np.hypot(max(abs(params[f"c{m}"].lo or 0), abs(params[f"c{m}"].hi or 0)),
                                     max(abs(params[f"s{m}"].lo or 0), abs(params[f"s{m}"].hi or 0)))
                            for m in x.ms)
                if worst >= FOURIER_SUM_MAX:
                    raise ValueError(f"{c.name}: the Fourier-mode ranges allow sum a_m >= {FOURIER_SUM_MAX}")
                continue
            cons.append(x)
        new = replace(c, params=params, constraints=tuple(cons))  # options (bend_pa_ref, ...) carried over
        comps.append(new)
        if "position" in spec and "x0" in params:
            extra.append(within(new, (best["x0"], best["y0"]), float(spec["position"]), "arcsec"))
    # Extra constraints of the base model: kept unless replaced by a position range.
    replaced = {x.component for x in extra}
    extra += [x for x in base.constraints.values()
              if x.component not in replaced and not any(x is y for c in base.components for y in c.constraints)]
    return ModelParams(comps, ties=base.ties, constraints=extra)


def seed_config(fit_folder, ranges=None, overrides=None, points=None):
    """Band B's configuration from a fit folder of band A (forwardgal.history).

    Takes the folder's model.yaml (the configuration that was run; an old
    quasar: section becomes a point component), replaces its model by
    ``seed_params`` (band A's best values from result.yaml, with the ranges)
    and then replaces whole sections by ``overrides`` (e.g. the band B
    "data", "psf", "label"; a "mask" if it differs).

    Point components (e.g. the quasar) have locked positions: they are not
    seeded (no ranges) but carried over as they are, in their places, e.g.
    with their RA/Dec (the same sky position in every band). A point's own
    PSF belongs to band A, so it is removed (a warning; band B's global PSF
    is used) unless ``points`` gives a new one:
        points = {"QSO 1A": {"psf": {"file": ..., "reference": "centroid"},
                             "max_snap": None}}
    updates those keys of the named points. An old-style "quasar" entry in
    ``overrides`` replaces the point of the same name.

    Flux parameters ('flux' or 'mag') are removed: band A's flux is another
    filter's, so band B's fluxes are solved afresh (F3, 2026-11-22). Ranges
    on them are refused for the same reason.
    Records the seed in cfg["seed"] (fit folder, ranges, removed PSFs and fluxes).
    """
    from forwardgal.config import upgrade_config

    folder = Path(fit_folder)
    cfg = upgrade_config(read_yaml(folder / "model.yaml"))
    best = read_yaml(folder / "result.yaml")["values"]
    overrides = dict(overrides or {})

    # Take the point components out (in their places), seed the rest, put them back.
    comps = cfg["model"]["components"]
    kept = [(i, copy.deepcopy(c)) for i, c in enumerate(comps) if c["kind"] == "point"]
    names = {c["name"] for _, c in kept}
    on_flux = sorted(f"{c}.{k}" for c, r in (ranges or {}).items() for k in r if k in FLUX_KEYS)
    if on_flux:
        raise ValueError(f"no ranges for fluxes {on_flux}: band B's fluxes are solved afresh")
    locked = names & set(ranges or {})
    if locked:
        raise ValueError(f"no ranges for point components {sorted(locked)}: their positions are locked")
    rest = dict(cfg["model"], components=[c for c in comps if c["kind"] != "point"])
    model = model_to_dict(seed_params(rest, best, ranges))

    if "quasar" in overrides:  # the old way to give band B's quasar: replaces the point of that name
        new = upgrade_config({"quasar": overrides.pop("quasar"), "model": {"components": []}})
        q = new["model"]["components"][0]
        at = [k for k, (_, c) in enumerate(kept) if c["name"] == q["name"]]
        if at:
            kept[at[0]] = (kept[at[0]][0], q)
        else:
            kept.append((len(comps), q))
    dropped = []
    for _, c in kept:
        if "psf" in c:
            del c["psf"]
            dropped.append(c["name"])
    unknown = set(points or {}) - {c["name"] for _, c in kept}
    if unknown:
        raise ValueError(f"points: no point component(s) {sorted(unknown)}")
    for _, c in kept:
        c.update(copy.deepcopy((points or {}).get(c["name"], {})))
    dropped = [n for n in dropped if "psf" not in (points or {}).get(n, {})]
    if dropped:
        warnings.warn(f"seeding: the band-A PSF of point(s) {dropped} is not carried over; band B's global "
                      "PSF is used (give points={name: {'psf': ...}} for a PSF of their own)", stacklevel=2)
    for i, c in sorted(kept, key=lambda x: x[0]):
        model["components"].insert(min(i, len(model["components"])), c)
    reset = []
    for c in model["components"]:  # band A's fluxes are another filter's: solved afresh in band B
        for k in FLUX_KEYS:
            if c["params"].pop(k, None) is not None:
                reset.append(c["name"])
    cfg["model"] = model

    for k, v in overrides.items():
        cfg[k] = v
    cfg["seed"] = {"from_fit": str(folder.resolve()), "ranges": ranges or {}}
    if dropped:
        cfg["seed"]["point_psf_removed"] = dropped
    if reset:
        cfg["seed"]["fluxes_reset"] = reset
    return cfg


# Defaults of the GUI's "Use in band..." dialog (M14's example; user, 2026-11-23).
DEFAULT_RANGES = {"*": {"position": 0.02, "R_e": {"rel": 0.2}, "n": 0.5, "q": 0.05, "pa_deg": 10.0}}


def seed_into(fit_folder, target_cfg, target_dir, ranges=None):
    """Band B's configuration seeded from band A's fit (the GUI's "Use in band...").

    fit_folder : band A's fit (history folder)
    target_cfg : band B's own configuration (its model file), whose paths are
        relative to ``target_dir``
    Every section of band B is kept (data, PSF, mask, lens, cosmology,
    render, fit, label, band, history_dir: band B's set-up), with absolute
    paths; the model is band A's fit seeded with ``ranges`` (seed_config:
    fluxes solved afresh); band B's point components keep their own PSF and
    max_snap. Points of band A missing in band B are carried over as they are.
    """
    from forwardgal.config import absolute_paths, upgrade_config

    target = absolute_paths(upgrade_config(target_cfg), target_dir)
    overrides = {k: v for k, v in target.items() if k not in ("model", "seed")}
    a_points = {c["name"] for c in upgrade_config(read_yaml(Path(fit_folder) / "model.yaml"))["model"]["components"]
                if c["kind"] == "point"}
    points = {}
    for c in target.get("model", {}).get("components", []):
        if c["kind"] == "point" and c["name"] in a_points:
            points[c["name"]] = {k: copy.deepcopy(c[k]) for k in ("psf", "max_snap") if k in c}
    with warnings.catch_warnings():  # (a point without a PSF of its own in B: B's global PSF, as intended)
        warnings.simplefilter("ignore")
        return seed_config(fit_folder, ranges, overrides, points)
