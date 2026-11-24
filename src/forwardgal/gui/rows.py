"""What the components column shows (G2b): display rows from the spec. No Qt here.

Inputs are the model spec (forwardgal.edit), the current values of every
parameter (LiveModel.values: ties and fixed values resolved) and the solved
amplitudes (LiveResult.amplitudes). Outputs are plain ``Row`` objects that
the Qt panel (gui/components_panel.py) only has to lay out.

Conventions shown to the user: positions in arcsec, labelled beta (source
plane) or theta (image plane), +x east; PA in degrees east of north. Units
are in the row labels, e.g. "R_e (″)"; value fields hold plain numbers. An
(e1, e2) Sersic is shown as q and PA (derived, as in the mockup): its e1, e2
carry the joint bound |e| < 0.905, so they cannot be locked or bounded alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forwardgal.components import sersic_qpa
from forwardgal.edit import constraint_text
import math

PLANE_SYMBOL = {"source": "β", "image": "θ"}


@dataclass(frozen=True)
class Row:
    """One line of the parameter card."""

    key: str  # parameter name in the spec ("q", "pa_deg" also for a derived e1/e2 Sersic); "flux"
    label: str  # e.g. "β_x (″)", "n", "PA (°)": the unit, if any, is here
    text: str  # formatted value, a plain number, e.g. "+0.0310"
    constraint: str  # e.g. "U(0.5, 8)", "= bulge.x0 + 0", "|e| < 0.905 (q > 0.05)", "linear (NNLS)"
    locked: bool  # fixed in the fit
    lockable: bool  # may the user change the lock here?
    tooltip: str = ""  # why not, or what it means
    editable: bool = True  # may the user type a new value? (not: tied, point positions, flux)


def label(key, plane):
    """Display label of a parameter key, with its unit: "β_x (″)", "R_e (″)", "PA (°)"."""
    s = PLANE_SYMBOL[plane]
    return {"x0": f"{s}_x (″)", "y0": f"{s}_y (″)", "R_e": "R_e (″)", "n": "n", "q": "q",
            "pa_deg": "PA (°)", "flux": "flux", "mag": "mag (AB)"}.get(key, _mode_label(key))


def _mode_label(key):
    """a3 -> "a_3", phi3 -> "φ_3 (°)", b2 -> "b_2 (″)" (modes); anything else as it is."""
    import re

    m = re.fullmatch(r"(a|phi|b)(\d)", key)
    if not m:
        return key
    return {"a": f"a_{m[2]}", "phi": f"φ_{m[2]} (°)", "b": f"b_{m[2]} (″)"}[m[1]]


def fmt(key, v):
    """Value text, a plain number (the unit is in the label): positions with a sign."""
    if key in ("x0", "y0"):
        return f"{v:+.4f}"
    if key == "R_e":
        return f"{v:.4f}"
    if key == "pa_deg":
        return f"{v:.1f}"
    if key in ("n", "q", "mag"):
        return f"{v:.3f}"
    if key == "flux":
        return f"{v:.5g}"
    return f"{v:.4g}"


def _joint(spec, comp):
    """{param key: description} of the joint constraints on this component's parameters."""
    out = {}
    for c in list(comp.get("constraints", [])) + [c for c in spec.get("constraints", [])
                                                  if c.get("component") == comp["name"]]:
        if c["type"] == "position":
            for k in c["keys"]:
                out[k] = c["description"]
        elif c["type"] == "fourier_bound":
            for m in c["ms"]:
                out[f"c{m}"] = out[f"s{m}"] = f"Σ a_m < {c['limit']:g}"
    return out


def param_rows(spec, name, values, amplitudes=None):
    """Rows of component ``name``: its parameters, then its flux.

    spec : model dict; values : {param: value} of this component (resolved);
    amplitudes : {component: amplitude} or None (flux row shows "-").
    """
    comp = next(c for c in spec["components"] if c["name"] == name)
    params, plane, ties = comp["params"], comp["plane"], spec.get("ties", {})
    joint = _joint(spec, comp)
    rows = []
    for key, p in params.items():
        if key == "e1":  # an (e1, e2) Sersic: shown as q and PA, in e1's place (after R_e, n)
            q, pa = (float(x) for x in sersic_qpa(values))
            why = ("derived from (e1, e2), which carry the joint bound " + joint.get("e1", "|e| < 0.905")
                   + "; locking it, or setting a constraint, first turns the component into (q, PA) "
                   "parameters at the current shape (G2e)")
            # editable (set_ellipse); lockable: the window converts to (q, PA) first (edit.to_qpa)
            rows.append(Row("q", "q", fmt("q", q), joint.get("e1", ""), False, True, why))
            rows.append(Row("pa_deg", label("pa_deg", plane), fmt("pa_deg", pa % 180.0), "", False, True, why))
            continue
        if key == "e2":
            continue
        if key[0] == "c" and key[1:].isdigit() and f"s{key[1:]}" in params:  # a Fourier mode: a_m, phi_m
            m = int(key[1:])
            c, s_ = values[f"c{m}"], values[f"s{m}"]
            a, phi = float(np.hypot(c, s_)), float(np.degrees(np.arctan2(s_, c)) / m)
            why = ("derived from (c_m, s_m) = a_m (cos m φ_m, sin m φ_m), which carry the joint bound "
                   + joint.get(key, "Σ a_m < 0.95") + "; editable, not lockable")
            rows.append(Row(f"a{m}", f"a_{m}", f"{a:.4f}", joint.get(key, ""), False, False, why))
            rows.append(Row(f"phi{m}", f"φ_{m} (°)", f"{phi:.1f}", "", False, False, why))
            continue
        if key[0] == "s" and key[1:].isdigit() and f"c{key[1:]}" in params:
            continue
        full = f"{name}.{key}"
        locked = p.get("free", True) is False
        if full in ties:
            src, off = ties[full]
            rows.append(Row(key, label(key, plane), fmt(key, values[key]), f"= {src} {'+' if off >= 0 else '-'} "
                            f"{abs(off):g}", locked, False, "tied: follows its source parameter", editable=False))
        elif key in joint:
            rows.append(Row(key, label(key, plane), fmt(key, values[key]), joint[key], locked, False,
                            "part of a joint constraint: it must stay free and unbounded"))
        elif comp["kind"] == "point" and key in ("x0", "y0"):
            rows.append(Row(key, label(key, plane), fmt(key, values[key]), constraint_text(p), locked, False,
                            "a point's position is locked; moving it needs the model built again", editable=False))
        else:
            tip = _flux_tooltip(comp, key) if key in ("flux", "mag") else ""
            rows.append(Row(key, label(key, plane), fmt(key, values[key]), constraint_text(p), locked, True, tip))
    if not any(k in params for k in ("flux", "mag")):  # an old spec: the amplitude is solved, not a parameter
        a = None if amplitudes is None else amplitudes.get(name)
        unit = "per pixel" if comp["kind"] == "sky" else "total"
        rows.append(Row("amp", "flux", "-" if a is None else f"{a:.5g}", f"solved (NNLS), {unit}", False, False,
                        "solved for every model: not a fit parameter", editable=False))
    return rows


def _flux_tooltip(comp, key):
    """What a flux parameter means for this component (intrinsic or observed, units)."""
    if comp["kind"] == "sky":
        return "sky level per pixel" if key == "flux" else "sky surface brightness (mag/arcsec²; needs sky > 0)"
    which = "intrinsic (unlensed)" if comp["plane"] == "source" else "observed"
    return f"{which} total {'flux (pixel-sum units)' if key == 'flux' else 'AB magnitude'}"


def summary(comp, values):
    """Short text for the component list, e.g. "n 3.80", "point (image)", "linear"."""
    if comp["kind"] == "sersic":
        return f"n {values['n']:.2f} · {comp['plane']}"
    if comp["kind"] == "point":
        return f"point · {comp['plane']}"
    if "flux" in comp["params"]:
        return f"{values['flux']:.3g} /px"
    if "mag" in comp["params"]:
        return f"{values['mag']:.2f} mag/″²"
    return "solved"


def parse_value(text):
    """The number typed into a value field, as a float.

    The fields hold plain numbers (units are in the row labels). Accepts
    surrounding spaces and a leading "+"; e.g. "+0.0310" -> 0.031,
    " 0.25 " -> 0.25, "1e-3" -> 0.001. Raises ValueError (with the text in
    the message) for anything else: an empty field, words, units ("30°"),
    and the non-finite values nan and inf.
    """
    # TODO(Erik): write this (about 5 lines).
    #   - str.strip() removes surrounding spaces.
    #   - float("+0.031") works already, and raises ValueError for "abc" or "30°";
    #     but float("nan") and float("inf") also "work", so check the result with
    #     math.isfinite.
    # Check with:  pytest tests/test_gui_rows.py -q -k parse_value
    x = float(text.strip())
    if not math.isfinite(x):
        raise ValueError(f"{x} is not finite.")
    return x