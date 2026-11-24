"""Editing a model spec, one checked change at a time (G2a). No Qt here.

The model spec is the ``model:`` section of a model YAML as a dict
(forwardgal.serialize): {"components": [...], "ties": {...},
"constraints": [...]}, with each parameter a dict like
{"value": 4.0, "lo": 0.5, "hi": 8.0} (see serialize.param_from_dict).

Every function here returns a NEW spec (the input is never changed) and
checks it the way a fit would: build ModelParams from it and compute the
starting vector. If that fails, ``SpecError`` is raised and the caller
keeps its old spec, so an edit is applied completely or not at all.
"""

from __future__ import annotations

import copy
import numpy as np

from forwardgal.serialize import model_from_dict
from forwardgal.components import Q_MIN, qpa_from_e

FIELDS = ("value", "free", "lo", "hi", "prior_mu", "prior_sigma")


class SpecError(ValueError):
    """An edit would make the spec invalid; the message says why."""


def check(spec):
    """ModelParams of ``spec``, or SpecError if a fit could not start from it."""
    try:
        params = model_from_dict(spec)
        params.start()  # also checks the starting values against joint constraints
    except (ValueError, KeyError, TypeError) as e:
        raise SpecError(str(e)) from e
    return params


def _param(spec, name, key):
    comp = next((c for c in spec["components"] if c["name"] == name), None)
    if comp is None:
        raise KeyError(f"no component {name!r}")
    if key not in comp["params"]:
        raise KeyError(f"{name}: no parameter {key!r}")
    return comp["params"][key]


def set_param(spec, name, key, **fields):
    """Copy of ``spec`` with fields of parameter ``name.key`` changed, checked.

    fields : any of value, free, lo, hi, prior_mu, prior_sigma. None removes
        a field (no bound, no prior); "value" cannot be removed. Give
        prior_mu and prior_sigma together. free=False locks the parameter at
        its value (a fit does not vary it).
    Values are taken as given (no clipping): a value outside the bounds is
    a SpecError, so a typo in the table is reported, not silently changed.
    """
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise TypeError(f"unknown field(s) {sorted(unknown)}; allowed: {FIELDS}")
    spec = copy.deepcopy(spec)
    p = _param(spec, name, key)
    for f, x in fields.items():
        if x is None:
            if f == "value":
                raise SpecError(f"{name}.{key}: the value cannot be removed")
            p.pop(f, None)
        elif f == "free":
            p["free"] = bool(x)
        else:
            p[f] = float(x)
    if p.get("free") is True:  # the default: leave it out of the file
        del p["free"]
    check(spec)
    return spec


def set_tie(spec, target, source, offset=0.0):
    """Copy of ``spec`` with ``target`` tied to ``source`` + offset, checked.

    target, source : "component.param". source None removes the tie.
    A tied parameter's own value is ignored by the fit (it follows source);
    both must be in the same plane, and source must not itself be tied.
    """
    spec = copy.deepcopy(spec)
    ties = spec.setdefault("ties", {})
    if source is None:
        ties.pop(target, None)
    else:
        ties[target] = [source, float(offset)]
    check(spec)
    return spec


def new_name(spec, stem="sersic"):
    """First unused name stem1, stem2, ... (component names must be unique)."""
    used = {c["name"] for c in spec["components"]}
    k = 1
    while f"{stem}{k}" in used:
        k += 1
    return f"{stem}{k}"


def add_sersic(spec, name, plane, x0, y0, R_e, q=0.8, pa_deg=0.0, n=1.0):
    """Copy of ``spec`` with a new Sersic component appended, checked.

    plane : "source" (x0, y0 = beta) or "image" (x0, y0 = theta), arcsec,
        x east, y north; R_e in arcsec; pa_deg east of north.
    It gets the default constraints of components.sersic_component (R_e
    >= 0.01", and <= 2" in the source plane; 0.5 < n < 8; ellipticity as
    (e1, e2)) and NO flux parameter: the caller adds one (the GUI fits it
    once to the light the other components leave over). Values are not
    clipped: e.g. R_e outside its bounds is a SpecError.
    """
    from forwardgal.components import sersic_component
    from forwardgal.serialize import component_to_dict

    if plane not in ("source", "image"):
        raise SpecError(f"plane must be 'source' or 'image', got {plane!r}")
    if any(c["name"] == name for c in spec["components"]):
        raise SpecError(f"there is already a component {name!r}")
    spec = copy.deepcopy(spec)
    try:
        c = sersic_component(name, plane, float(x0), float(y0), float(R_e), float(n), float(q), float(pa_deg))
    except ValueError as e:  # e.g. a starting value outside its default bounds
        raise SpecError(f"{name}: {e}") from e
    spec["components"].append(component_to_dict(c))
    check(spec)
    return spec


def to_qpa(spec, name):
    """Copy of ``spec`` with Sersic ``name`` in (q, pa_deg) instead of (e1, e2), at the same shape, checked.

    Locking q or PA needs this: in (e1, e2) neither is a parameter of its own.
      q, pa = components.qpa_from_e(e1, e2)   (PA east of north, in (-90, 90])
    The new parameters replace e1 and e2 in the same place in the params dict
    (after n, before any modes or the flux), with the bounds of
    sersic_component(..., ellipticity="qpa"):
      "q":      {"value": q,          "lo": Q_MIN (0.05), "hi": 1.0}
      "pa_deg": {"value": pa % 180.0, "lo": -1e-6,        "hi": 180.0}
    Bounds are strict (lo < value < hi), so a round component (q = 1) or one at
    the edge (q = 0.05) is moved just inside, by 1e-6. The component's joint
    constraint on (e1, e2) (the |e| < 0.905 bound: the entry of
    comp["constraints"] whose "keys" are ["e1", "e2"]) is removed: the q
    bounds replace it. A component already in (q, pa_deg): returned unchanged.
    Example: e1 = 0.125, e2 = 0.2165 (q = 0.6, PA = 30) -> q 0.6, pa_deg 30.
    """
    # TODO(Erik): write this (about 10 lines).
    #   - Copy first; find the component dict (as in _param: next(...), or a KeyError).
    #   - Nothing to do if "e1" is not in its params.
    #   - from forwardgal.components import Q_MIN, qpa_from_e ; float() its results (they are JAX numbers).
    #   - Keep the order: build a NEW params dict in a loop over the old items, putting "q" where
    #     "e1" was and "pa_deg" where "e2" was (assigning comp["params"] = new at the end).
    #   - Drop the (e1, e2) constraint: keep [k for k in comp.get("constraints", []) if "e1" not in k["keys"]].
    #   - End with check(spec) and return spec.
    # Check with:  pytest tests/test_edit.py -q -k qpa
    spec = copy.deepcopy(spec)
    comp = next((c for c in spec["components"] if c["name"] == name), None)
    if comp is None:
        raise KeyError(f"no component {name!r}")
    params = comp["params"]
    if "e1" not in params:
        return spec
    q, pa = (float(x) for x in qpa_from_e(params["e1"]["value"], params["e2"]["value"]))
    q = min(max(q, Q_MIN + 1e-6), 1.0 - 1e-6)
    new = {}
    for key, p in params.items():
        if key == "e1":
            new["q"] = {"value": q, "lo": Q_MIN, "hi": 1.0}
        elif key == "e2":
            new["pa_deg"] = {"value": pa % 180.0, "lo": -1e-6, "hi": 180.0}
        else:
            new[key] = p
    comp["params"] = new
    comp["constraints"] = [k for k in comp.get("constraints", []) if "e1" not in k["keys"]]
    check(spec)
    return spec

def remove_component(spec, name):
    """Copy of ``spec`` without component ``name``, and without any tie to or from it, checked.

    A tie "target": [source, offset] involves component X if target or
    source starts with "X." (e.g. "disk.x0"). Example: removing "clump" from
    ties {"clump.x0": ["disk.x0", 0.0], "bulge.n": ["disk.n", 0.0]} keeps
    only the bulge.n tie. Unknown name: KeyError.
    """
    if not any(c["name"] == name for c in spec["components"]):
        raise KeyError(f"no component {name!r}")
    spec = copy.deepcopy(spec)
    spec["components"] = [c for c in spec["components"] if c["name"] != name]
    ties = spec.get("ties", {})
    spec["ties"] = {t: v for t, v in ties.items()
                    if t.split(".")[0] != name and v[0].split(".")[0] != name}
    check(spec)
    return spec

def constraint_text(p):
    """Short text for the table's constraint column, from a parameter dict p.

    p : {"value": ..., optional "lo", "hi", "prior_mu", "prior_sigma", "free"}
    Bounds:  both -> "U(lo, hi)";  lo only -> "> lo";  hi only -> "< hi".
    Prior:   "N(mu, sigma)".
    Both present: bounds first, then the prior, joined by ", ".
    Neither: "".  ("free" is shown by the lock column, not here.)
    Numbers in the shortest form, f"{x:g}": 8.0 -> "8", 0.01 -> "0.01".
    Example: {"value": 4, "lo": 0.5, "hi": 8.0, "prior_mu": 4, "prior_sigma": 1}
             -> "U(0.5, 8), N(4, 1)"
    """
    parts = []
    lo, hi = p.get("lo"), p.get("hi")
    if lo is not None and hi is not None:
        parts.append(f"U({lo:g}, {hi:g})")
    elif lo is not None:
        parts.append(f"> {lo:g}")
    elif hi is not None:
        parts.append(f"< {hi:g}")
    mu, sigma = p.get("prior_mu"), p.get("prior_sigma")
    if mu is not None and sigma is not None:
        parts.append(f"N({mu:g}, {sigma:g})")
    return ", ".join(parts)


def tie_candidates(spec, name, key):
    """Parameters that ``name.key`` may be tied to: a list of "component.key" strings.

    A candidate is the SAME parameter key (x0 to x0: an offset in the same
    units) of ANOTHER component in the SAME plane (a tie must not mix source-
    and image-plane positions), which is not itself tied (no tie chains).
    Components are listed in spec order. Example, for the bulge/disk/QSO
    spec: tie_candidates(spec, "disk", "x0") -> ["bulge.x0"]; the QSO (image
    plane) is not a candidate, and nothing is if bulge.x0 is tied itself.
    """
    plane = next((c["plane"] for c in spec["components"] if c["name"] == name), None)
    if plane is None:
        raise KeyError(f"no component {name!r}")
    tied = set(spec.get("ties", {}))
    return [s for c in spec["components"]
            if c["name"] != name and c["plane"] == plane and key in c["params"]
            and (s := f"{c['name']}.{key}") not in tied]


def hold_others(spec, names):
    """Copy of ``spec`` for fitting only the components ``names``: every other one held at its values, checked.

    "Fit selected" (G4): each parameter of the other components is locked
    (free: False), except a tied one (it follows its source anyway). Their
    joint constraints (the (e1, e2) |e| bound, a Fourier bound; also
    model-level ones naming them) are dropped: those need free parameters,
    and fixed values already satisfy them. Only for the fit: the saved model
    keeps the user's settings.
    """
    names = set(names)
    unknown = names - {c["name"] for c in spec["components"]}
    if unknown:
        raise KeyError(f"no component(s) {sorted(unknown)}")
    spec = copy.deepcopy(spec)
    ties = spec.get("ties", {})
    for comp in spec["components"]:
        if comp["name"] in names:
            continue
        for key, p in comp["params"].items():
            if f"{comp['name']}.{key}" not in ties:
                p["free"] = False
        comp.pop("constraints", None)
    spec["constraints"] = [c for c in spec.get("constraints", []) if c.get("component") not in
                           {x["name"] for x in spec["components"]} - names]
    check(spec)
    return spec


# --- Fourier and bending modes (G6a) -------------------------------------------------------


def _component(spec, name):
    comp = next((c for c in spec["components"] if c["name"] == name), None)
    if comp is None:
        raise KeyError(f"no component {name!r}")
    return comp


def _insert_before_flux(params, new):
    """params with the items of ``new`` added before the flux parameter (the card shows the flux last)."""
    out = {k: v for k, v in params.items() if k not in ("flux", "mag")}
    out.update(new)
    out.update({k: v for k, v in params.items() if k in ("flux", "mag")})
    return out


def _fourier_bound(comp):
    return next((k for k in comp.get("constraints", []) if k.get("type") == "fourier_bound"), None)


def add_mode(spec, name, kind, m):
    """Copy of ``spec`` with a Fourier (kind "fourier", m in 1, 3, 4) or bending ("bending", m in 2..5)
    mode added to Sersic ``name`` at zero amplitude (the shape does not change), checked.

    Fourier: parameters c_m, s_m (Cartesian; a_m = phi_m = 0), and m joins the
    component's joint bound sum a_m < 0.95 (created if needed). Bending: b_m
    (arcsec, unbounded); the first bending mode fixes the bend frame at the
    component's current PA (bend_pa_ref; the seam lies 90 deg from it).
    """
    from forwardgal.components import BEND_M, FOURIER_M, FOURIER_SUM_MAX, sersic_qpa

    spec = copy.deepcopy(spec)
    comp = _component(spec, name)
    if comp["kind"] != "sersic":
        raise SpecError(f"{name}: modes are for Sersic components")
    p = comp["params"]
    if kind == "fourier":
        if m not in FOURIER_M:
            raise SpecError(f"Fourier mode m = {m} is not allowed (allowed: {FOURIER_M})")
        if f"a{m}" in p or any(k.startswith("a") and k[1:].isdigit() for k in p):
            raise SpecError(f"{name}: its modes are in (a_m, phi_m) form; add modes in the YAML file")
        if f"c{m}" in p:
            raise SpecError(f"{name} already has Fourier mode m = {m}")
        comp["params"] = _insert_before_flux(p, {f"c{m}": {"value": 0.0}, f"s{m}": {"value": 0.0}})
        fb = _fourier_bound(comp)
        if fb is None:
            comp.setdefault("constraints", []).append({"type": "fourier_bound", "component": name, "ms": [m],
                                                       "limit": FOURIER_SUM_MAX, "delta": 0.01})
        else:
            fb["ms"] = sorted(set(fb["ms"]) | {m})
    elif kind == "bending":
        if m not in BEND_M:
            raise SpecError(f"bending mode m = {m} is not allowed (allowed: {BEND_M})")
        if f"b{m}" in p:
            raise SpecError(f"{name} already has bending mode m = {m}")
        if not any(k[0] == "b" and k[1:].isdigit() for k in p):  # the first one: the bend frame
            values = check(spec).unpack(check(spec).start())[name]
            comp["bend_pa_ref"] = float(sersic_qpa(values)[1]) % 180.0
        comp["params"] = _insert_before_flux(p, {f"b{m}": {"value": 0.0}})
    else:
        raise SpecError(f"kind must be 'fourier' or 'bending', got {kind!r}")
    check(spec)
    return spec


def remove_mode(spec, name, kind, m):
    """Copy of ``spec`` without that mode of ``name`` (and its ties, and its place in the Fourier bound), checked."""
    spec = copy.deepcopy(spec)
    comp = _component(spec, name)
    keys = [f"c{m}", f"s{m}"] if kind == "fourier" else [f"b{m}"]
    if not all(k in comp["params"] for k in keys):
        raise SpecError(f"{name} has no {kind} mode m = {m}")
    for k in keys:
        del comp["params"][k]
    gone = {f"{name}.{k}" for k in keys}
    spec["ties"] = {t: v for t, v in spec.get("ties", {}).items() if t not in gone and v[0] not in gone}
    if kind == "fourier":
        fb = _fourier_bound(comp)
        fb["ms"] = [x for x in fb["ms"] if x != m]
        if not fb["ms"]:
            comp["constraints"] = [k for k in comp["constraints"] if k is not fb]
    elif not any(k[0] == "b" and k[1:].isdigit() for k in comp["params"]):
        comp.pop("bend_pa_ref", None)
    check(spec)
    return spec


def set_mode(spec, name, m, a=None, phi_deg=None):
    """Copy of ``spec`` with Fourier mode m of ``name`` set to amplitude ``a`` and phase ``phi_deg``, checked.

    The card shows a_m and phi_m (GALFIT's amplitude and phase); the parameters
    are c_m = a_m cos(m phi_m), s_m = a_m sin(m phi_m) (phi_m in degrees; the
    cosine and sine take radians). A value not given (None) stays as it is
    now: a_m = sqrt(c_m^2 + s_m^2), phi_m = atan2(s_m, c_m) / m (in degrees).
    Example: m = 1, a = 0.04, phi_deg = 12 -> c1 = 0.0391, s1 = 0.0083.
    """
    # TODO(Erik): write this (about 8 lines).
    #   - Copy first, and find the component with _component(spec, name) (above).
    #   - c, s = its params[f"c{m}"]["value"], params[f"s{m}"]["value"] (a KeyError if it has no such mode).
    #   - If a is None: a = the current amplitude; if phi_deg is None: the current phase (np.hypot, np.arctan2,
    #     np.degrees / np.radians; numpy is not imported in this file yet: add "import numpy as np" at the top).
    #   - Write the new c and s (as plain floats) into the two params, then check(spec) and return spec.
    # Check with:  pytest tests/test_edit.py -q -k set_mode
    spec = copy.deepcopy(spec)
    params = _component(spec, name)["params"]
    c = params[f"c{m}"]["value"]
    s = params[f"s{m}"]["value"]
    if a is None:
        a = np.hypot(c, s)
    if phi_deg is None:
        phi_deg = np.degrees(np.arctan2(s, c))/ m
    params[f"c{m}"]["value"] = float(a * np.cos(np.radians(m * phi_deg)))
    params[f"s{m}"]["value"] = float(a * np.sin(np.radians(m * phi_deg)))
    check(spec)
    return spec
