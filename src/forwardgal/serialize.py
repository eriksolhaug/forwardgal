"""Model parameters <-> plain dictionaries (M13a) <-> YAML files (M13b).

The dictionaries contain only str, float, bool, list and dict (and None is
never written), so they can be written to YAML / JSON directly. Floats are
kept as Python floats (YAML writes them with repr, i.e. exactly), so
``model_from_dict(model_to_dict(p))`` rebuilds ``p`` exactly.

Format (defaults are omitted: a parameter is free, unbounded and without a
prior unless the dictionary says otherwise)::

    components:
      - name: disk
        kind: sersic                 # sersic | point | sky
        plane: source                # source | image
        bend_pa_ref: 16.0            # only for components with bending modes
        params:
          x0:  {value: -5.7594}
          R_e: {value: 0.393, lo: 0.01}
          n:   {value: 0.55, lo: 0.5, hi: 8.0}
          q:   {value: 0.75, free: false}
          e1:  {value: 0.1, prior_mu: 0.0, prior_sigma: 0.4}
        constraints:                 # joint constraints attached to the component
          - {type: position, ref: [0.0, 0.0], radius: 0.905, matrix: [[1.0, 0.0], [0.0, 1.0]],
             mode: bounds, keys: [e1, e2], description: "..."}
          - {type: fourier_bound, ms: [1, 3, 4], limit: 0.95, delta: 0.01}
    ties: {"disk.x0": ["bulge.x0", 0.0]}     # target: [source, offset]
    constraints: []                          # extra joint constraints (e.g. "within N")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from forwardgal.components import Component, FourierBound, ModelParams, PositionConstraint
from forwardgal.params import Param

PARAM_FIELDS = ("value", "free", "lo", "hi", "prior_mu", "prior_sigma")
PARAM_DEFAULTS = {"free": True, "lo": None, "hi": None, "prior_mu": None, "prior_sigma": None}


def param_to_dict(p):
    """Param -> {"value": ..., and only the fields that differ from the defaults}."""
    out = {"value": float(p.value)}
    for f in PARAM_FIELDS[1:]:
        v = getattr(p, f)
        if v != PARAM_DEFAULTS[f]:
            out[f] = bool(v) if f == "free" else float(v)
    return out


def param_from_dict(d):
    """{"value": ..., optional "free", "lo", "hi", "prior_mu", "prior_sigma"} -> Param.

    The inverse of ``param_to_dict``: missing fields take the defaults
    (free = True, no bounds, no prior). Unknown keys are an error (a typo in a
    YAML file, e.g. "hi " or "upper", must not be silently ignored).
    """
    # TODO(Erik): build and return the Param.
    #   - Check for unknown keys first: compare set(d) with PARAM_FIELDS and
    #     raise ValueError naming the unknown key(s).
    #   - "value" is required (a KeyError is fine, but a clear ValueError is nicer).
    #   - The other fields come from d if present, else from PARAM_DEFAULTS.
    #   - Param itself checks bounds and priors (see params.Param.__post_init__).
    # Check your version with:  pytest tests/test_serialize.py -q
    for k in d:
        if k not in PARAM_FIELDS:
            raise ValueError(f"{k} not in parameter list.")
    if "value" not in d.keys():
        raise ValueError("'value' is needed in parameter list.")
    return Param(value=d["value"], free=d.get("free", PARAM_DEFAULTS["free"]), lo=d.get("lo", PARAM_DEFAULTS["lo"]), hi=d.get("hi", PARAM_DEFAULTS["hi"]), prior_mu=d.get("prior_mu", PARAM_DEFAULTS["prior_mu"]), prior_sigma=d.get("prior_sigma", PARAM_DEFAULTS["prior_sigma"]))

def _constraint_to_dict(c):
    if isinstance(c, FourierBound):
        return {"type": "fourier_bound", "component": c.component, "ms": [int(m) for m in c.ms],
                "limit": float(c.limit), "delta": float(c.delta)}
    if isinstance(c, PositionConstraint):
        return {"type": "position", "component": c.component, "ref": [float(x) for x in c.ref],
                "radius": float(c.radius), "matrix": [[float(x) for x in row] for row in np.asarray(c.matrix)],
                "mode": c.mode, "description": c.description, "keys": list(c.keys)}
    raise TypeError(f"cannot serialize constraint {c!r}")


def _constraint_from_dict(d):
    d = dict(d)
    kind = d.pop("type")
    if kind == "fourier_bound":
        return FourierBound(d["component"], tuple(d["ms"]), d["limit"], d["delta"])
    if kind == "position":
        return PositionConstraint(d["component"], tuple(d["ref"]), d["radius"], np.array(d["matrix"], dtype=float),
                                  d["mode"], d["description"], tuple(d["keys"]))
    raise ValueError(f"unknown constraint type {kind!r}")


def component_to_dict(c):
    out = {"name": c.name, "kind": c.kind, "plane": c.plane,
           "params": {k: param_to_dict(c.params[k]) for k in c.names}}
    if c.bends:
        out["bend_pa_ref"] = float(c.bend_pa_ref)
    if c.kind == "point":
        out["max_snap"] = None if c.max_snap is None else float(c.max_snap)
        if c.psf is not None:
            out["psf"] = dict(c.psf)
    if c.constraints:
        out["constraints"] = [_constraint_to_dict(x) for x in c.constraints]
    return out


def component_from_dict(d):
    if "ra" in d or "dec" in d:
        raise ValueError(f"component {d.get('name')!r}: an ra/dec position is converted to x0, y0 by "
                         "config.build (load the whole model file, or give x0, y0 in arcsec)")
    unknown = set(d) - {"name", "kind", "plane", "params", "bend_pa_ref", "constraints", "max_snap", "psf"}
    if unknown:
        raise ValueError(f"component {d.get('name')!r}: unknown key(s) {sorted(unknown)}")
    return Component(name=d["name"], kind=d["kind"], plane=d["plane"],
                     params={k: param_from_dict(v) for k, v in d["params"].items()},
                     constraints=tuple(_constraint_from_dict(x) for x in d.get("constraints", [])),
                     bend_pa_ref=float(d.get("bend_pa_ref", 0.0)),
                     max_snap=d.get("max_snap", 0.1), psf=d.get("psf"))


def model_to_dict(params):
    """ModelParams -> plain dictionary (see the module docstring)."""
    attached = {id(x) for c in params.components for x in c.constraints}
    extra = [x for x in list(params.constraints.values()) + list(params.fourier_bounds.values())
             if id(x) not in attached]
    return {"components": [component_to_dict(c) for c in params.components],
            "ties": {t: [s, float(o)] for t, (s, o) in params.ties.items()},
            "constraints": [_constraint_to_dict(x) for x in extra]}


def model_from_dict(d):
    """Plain dictionary -> ModelParams (inverse of ``model_to_dict``)."""
    unknown = set(d) - {"components", "ties", "constraints"}
    if unknown:
        raise ValueError(f"model: unknown key(s) {sorted(unknown)}")
    return ModelParams([component_from_dict(c) for c in d["components"]],
                       ties={t: (s, float(o)) for t, (s, o) in d.get("ties", {}).items()},
                       constraints=[_constraint_from_dict(x) for x in d.get("constraints", [])])


# --- YAML files (M13b) --------------------------------------------------------------


def write_yaml(data, path, header=None):
    """Write a plain dictionary to a YAML file (safe dumper: plain data only).

    Key order is kept (sort_keys=False) and floats are written with Python's
    repr, so ``read_yaml`` gives back exactly the same numbers. ``header``: an
    optional comment (str, may have several lines) written at the top.
    """
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=110, allow_unicode=True)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    Path(path).write_text(text)


def read_yaml(path):
    """Read a YAML file written by ``write_yaml`` (or by hand) -> plain dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)
