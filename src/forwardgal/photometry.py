"""Flux <-> AB magnitude for components' flux parameters (F1, 2026-11-22).

Flux F is the sum of pixel values (data units; e.g. MJy/sr x pixels for
JWST i2d mosaics), so with the image's zeropoint ZP (forwardgal.data)

    m = -2.5 log10(F) + ZP,        F = 10^(-0.4 (m - ZP))           (1)

as in GALFIT. A source-plane component's flux is intrinsic (unlensed); an
image-plane component's is observed.

The sky is a level s per pixel; as a magnitude it is a surface brightness
mu_sky (mag arcsec^-2): one arcsec^2 holds s / A_pix of pixel sum, with
A_pix = pixscale^2 the pixel area in arcsec^2, so

    mu_sky = -2.5 log10(s / A_pix) + ZP,   s = A_pix 10^(-0.4 (mu_sky - ZP))   (2)

which needs s > 0 (background-subtracted images often have s ~ 0 or < 0).
"""

from __future__ import annotations

import copy
import math


def mag_to_flux(m, zp):
    """Total flux from an AB magnitude, eq. (1)."""
    return 10.0 ** (-0.4 * (m - zp))


def flux_to_mag(f, zp):
    """AB magnitude of a total flux f > 0, eq. (1)."""
    if not f > 0:
        raise ValueError(f"a magnitude needs a positive flux, got {f:g}")
    return -2.5 * math.log10(f) + zp


def sb_to_sky(mu, zp, pixscale):
    """Sky level per pixel from a surface brightness mu (mag arcsec^-2), eq. (2)."""
    return pixscale**2 * 10.0 ** (-0.4 * (mu - zp))


def sky_to_sb(s, zp, pixscale):
    """Surface brightness (mag arcsec^-2) of a sky level s > 0 per pixel, eq. (2)."""
    if not s > 0:
        raise ValueError(f"a sky surface brightness needs a positive sky level, got {s:g} per pixel")
    return -2.5 * math.log10(s / pixscale**2) + zp


def converter(kind, zp, pixscale):
    """(to_mag, to_flux) functions for a component kind ("sky" uses eq. 2, the others eq. 1)."""
    if kind == "sky":
        return (lambda f: sky_to_sb(f, zp, pixscale)), (lambda m: sb_to_sky(m, zp, pixscale))
    return (lambda f: flux_to_mag(f, zp)), (lambda m: mag_to_flux(m, zp))


def convert_param(p, to, kind, zp, pixscale):
    """A flux parameter's dict converted to ``to`` ("flux" or "mag"); returns (dict, notes).

    p : {"value": ..., optional "free", "lo", "hi", "prior_mu", "prior_sigma"},
        in the other unit.
    Value and bounds convert exactly; since m decreases as F grows, the
    bounds swap ends (the faint magnitude limit is the LOWER flux bound). A
    flux lower bound of 0 (or below) becomes "no faint magnitude limit".
    A Gaussian prior does not stay Gaussian: it is converted to first order,
    sigma_m = 2.5 / ln(10) sigma_F / F = 1.0857 sigma_F / F (and back), and
    a note says so. ``notes`` is a list of such remarks (empty if none).
    """
    if to not in ("flux", "mag"):
        raise ValueError(f"to must be 'flux' or 'mag', got {to!r}")
    to_mag, to_flux = converter(kind, zp, pixscale)
    f = to_mag if to == "mag" else to_flux
    out, notes = {"value": float(f(p["value"]))}, []
    if "free" in p:
        out["free"] = p["free"]
    lo, hi = p.get("lo"), p.get("hi")
    if to == "mag":  # flux (lo, hi) -> mag (to_mag(hi), to_mag(lo))
        if hi is not None:
            out["lo"] = float(to_mag(hi))
        if lo is not None and lo > 0:
            out["hi"] = float(to_mag(lo))
        elif lo is not None:
            notes.append(f"flux lower bound {lo:g} has no magnitude: no faint limit")
    else:  # mag (lo, hi) -> flux (to_flux(hi), to_flux(lo))
        if hi is not None:
            out["lo"] = float(to_flux(hi))
        if lo is not None:
            out["hi"] = float(to_flux(lo))
    if p.get("prior_mu") is not None:
        mu, s = p["prior_mu"], p["prior_sigma"]
        k = 2.5 / math.log(10.0)  # dm = -k dF / F
        if to == "mag":
            out["prior_mu"], out["prior_sigma"] = float(to_mag(mu)), float(k * s / mu)
        else:
            f_mu = float(to_flux(mu))
            out["prior_mu"], out["prior_sigma"] = f_mu, float(s * f_mu / k)
        notes.append("the Gaussian prior was converted to first order (exact only for a small width)")
    return out, notes


# --- flux parameters in a model spec (F4a) -----------------------------------------


def _floor(amplitudes):
    """A faint starting flux for one that NNLS pinned at 0: 1e-4 of the brightest (or 1e-6)."""
    pos = [a for a in amplitudes.values() if a > 0]
    return 1e-4 * max(pos) if pos else 1e-6


def _in_unit(a, key, kind, zp, pixscale, floor):
    """Amplitude a as a value of flux parameter ``key`` ('flux' or 'mag'); (value, note or None)."""
    if key == "flux":
        return float(a), None
    if a > 0:
        return float(sky_to_sb(a, zp, pixscale) if kind == "sky" else flux_to_mag(a, zp)), None
    a = floor
    return float(sky_to_sb(a, zp, pixscale) if kind == "sky" else flux_to_mag(a, zp)), "no magnitude for flux <= 0"


def add_flux_parameters(spec, amplitudes, zeropoint, pixscale, unit=None):
    """Copy of ``spec`` where every component without a flux parameter gets one.

    amplitudes : {name: amplitude} (e.g. one NNLS solve: LiveModel.render)
    The sky gets 'flux' (per pixel, unbounded: it may be negative); others
    get 'mag' if there is a zeropoint, else 'flux' with a lower bound of 0.
    unit "flux" gives the others 'flux' even with a zeropoint (the GUI's
    Settings choice); "mag" without a zeropoint is an error.
    A flux NNLS pinned at 0 starts at a faint floor (1e-4 of the brightest).
    Returns (new spec, names given a flux parameter).
    """
    if unit not in (None, "flux", "mag"):
        raise ValueError(f"unit must be None, 'flux' or 'mag', got {unit!r}")
    if unit == "mag" and zeropoint is None:
        raise ValueError("magnitudes need the image's zeropoint")
    spec = copy.deepcopy(spec)
    floor, added = _floor(amplitudes), []
    for c in spec["components"]:
        p = c["params"]
        if any(k in p for k in ("flux", "mag")) or c["name"] not in amplitudes:
            continue
        a = float(amplitudes[c["name"]])
        if c["kind"] == "sky":
            p["flux"] = {"value": a}
        elif zeropoint is not None and unit != "flux":
            p["mag"] = {"value": _in_unit(a, "mag", c["kind"], zeropoint, pixscale, floor)[0]}
        else:
            p["flux"] = {"value": max(a, floor), "lo": 0.0}
        added.append(c["name"])
    return spec, added


def set_fluxes(spec, amplitudes, zeropoint, pixscale):
    """Copy of ``spec`` with the flux parameters of ``amplitudes`` ({name: amplitude}) set to them.

    Each value is written in its parameter's own unit ('flux' or 'mag'). A
    value outside the parameter's bounds is moved just inside them (a note
    says so); the spec is checked (forwardgal.edit). Returns (spec, notes).
    """
    from forwardgal.edit import check

    spec = copy.deepcopy(spec)
    floor, notes = _floor(amplitudes), []
    for c in spec["components"]:
        if c["name"] not in amplitudes:
            continue
        key = next((k for k in ("flux", "mag") if k in c["params"]), None)
        if key is None:
            continue
        p = c["params"][key]
        a = float(amplitudes[c["name"]])
        if key == "mag" and c["kind"] == "sky" and a <= 0:
            notes.append(f"{c['name']}: sky {a:.3g} <= 0 has no surface brightness; kept {p['value']:g}")
            continue
        x, note = _in_unit(a, key, c["kind"], zeropoint, pixscale, floor)
        if note:
            notes.append(f"{c['name']}: {note}; started at a faint value")
        lo, hi = p.get("lo"), p.get("hi")
        eps = 1e-9
        if lo is not None and x <= lo:
            x = lo + eps * max(1.0, abs(lo))
            notes.append(f"{c['name']}.{key}: at its lower bound {lo:g}")
        if hi is not None and x >= hi:
            x = hi - eps * max(1.0, abs(hi))
            notes.append(f"{c['name']}.{key}: at its upper bound {hi:g}")
        p["value"] = float(x)
    check(spec)
    return spec, notes


def convert_fluxes(spec, to, zeropoint, pixscale):
    """Copy of ``spec`` with every flux parameter converted to ``to`` ("flux" or "mag"): the Settings menu.

    Uses convert_param (value and bounds exact; priors first order). A sky
    that is <= 0 has no surface brightness: it stays in flux (a note says
    so). Flux parameters in a tie are refused (an offset in magnitudes is a
    flux RATIO: the tie would change meaning). Returns (spec, notes).
    """
    from forwardgal.edit import SpecError, check

    if to == "mag" and zeropoint is None:
        raise SpecError("magnitudes need the image's zeropoint")
    tied = sorted(k for t, (s, _) in spec.get("ties", {}).items() for k in (t, s)
                  if k.split(".", 1)[1] in ("flux", "mag"))
    if tied:
        raise SpecError(f"untie {tied} first: a tie's offset changes meaning between flux and magnitude")
    spec = copy.deepcopy(spec)
    notes = []
    for c in spec["components"]:
        p = c["params"]
        key = next((k for k in ("flux", "mag") if k in p), None)
        if key is None or key == to:
            continue
        try:
            new, n = convert_param(p[key], to, c["kind"], zeropoint, pixscale)
        except ValueError as e:
            notes.append(f"{c['name']}: kept in {key} ({e})")
            continue
        del p[key]
        p[to] = new
        notes += [f"{c['name']}: {x}" for x in n]
    check(spec)
    return spec, notes
