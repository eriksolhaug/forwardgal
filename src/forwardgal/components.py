"""Model components and the vector of free parameters.

A component has a name, a kind ("sersic", "point", "sky"), a plane
("source": lensed, parameters in beta; "image": unlensed, parameters in theta)
and its SHAPE parameters as ``Param``s. Amplitudes (brightness) are not here:
they are linear and solved directly (forwardgal.likelihood).

``ModelParams`` collects all free, untied parameters of all components into
one flat array u (unconstrained, see forwardgal.params), in a fixed order,
and converts back (``unpack``). Ties make one parameter equal another
(+ an offset); tied parameters are not in u.

Positions are arcsec, +x east, +y north; PA in degrees east of north.

Ellipticity of a Sersic: by default the components (e1, e2) (user's choice,
2026-11-16), with phi = PA (east of north) and eps = (1 - q)/(1 + q):

    e1 = eps cos 2 phi,  e2 = eps sin 2 phi;   q = (1 - eps)/(1 + eps),
    phi = atan2(e2, e1) / 2,  eps = sqrt(e1^2 + e2^2).

Round is e = 0 (no singular PA), PA and PA + 180 are the same point (no
0/180 seam). The bound eps < EPS_MAX (q > Q_MIN) is enforced jointly on
(e1, e2) by the same smooth 2-D map as a "within" constraint. With the
major-axis R_e convention the profile depends on |e|, so it has a mild kink
exactly at round shapes. ellipticity="qpa" keeps (q, pa_deg) instead.

Fourier modes of a Sersic (M12, user's choices 2026-11-16): any of m = 1, 3,
4, as Cartesian pairs (default) c_m = a_m cos(m phi_m), s_m = a_m sin(m phi_m)
(a_m, phi_m = GALFIT's amplitude and phase; reported as derived values),
with the joint bound sum_m a_m < FOURIER_SUM_MAX (keeps g > 0), see
``FourierBound``. mode_style="aphi" fits (a_m, phi_m) directly with
|a_m| < FOURIER_SUM_MAX / K (K = number of modes).

Bending modes of a Sersic (M12f, user's choices 2026-11-16): any of m = 2..5
(B1 is a pure shear, i.e. only a different q and PA, like Fourier m = 2),
parameters b2..b5 in ARCSEC (ridge shift at |u| = R_e), unbounded. The seam
of the bend frame is placed 90 deg from the starting PA (Component.bend_pa_ref
= starting PA); amplitudes in GALFIT's convention are reported as derived
b{m}_galfit.

Spiral rotation of a Sersic (M12e): PAUSED (user, 2026-11-16); its code is
commented out with "# [M12e paused]" (bounds and lessons in MILESTONES.md).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
import numpy as np

from forwardgal.params import Param
# [M12e paused] from forwardgal.profiles.sersic import CartesianModes, Spiral
from forwardgal.profiles.sersic import Bending, CartesianModes  # [M12e stand-in]

SHAPE_PARAMS = {  # allowed parameter sets per kind, in their canonical order
    "sersic": (("x0", "y0", "R_e", "n", "e1", "e2"), ("x0", "y0", "R_e", "n", "q", "pa_deg")),
    "point": (("x0", "y0"),),
    "sky": ((),),
}
FLUX_KEYS = ("flux", "mag")  # a component's flux parameter (at most one; listed last): see photometry.py
PLANES = ("source", "image")
R_E_MIN = 0.01  # arcsec: default lower bound on Sersic R_e
R_E_MAX_SOURCE = 2.0  # arcsec: default upper bound on a SOURCE-plane Sersic's (intrinsic) R_e (Erik,
#   2026-11-22: a 1A fit found a 9.5" disk filling the mask, degenerate with the sky and the quasar).
#   Image-plane Sersics (e.g. cluster members) have no default upper bound.
Q_MIN = 0.05  # default lower bound on q (both ellipticity parametrizations)
EPS_MAX = (1.0 - Q_MIN) / (1.0 + Q_MIN)  # = 0.905: |e| bound equivalent to q > Q_MIN
FOURIER_M = (1, 3, 4)  # allowed Fourier modes (m = 2 duplicates q, PA)
FOURIER_SUM_MAX = 0.95  # sum_m a_m < this, so g = 1 + sum a_m cos(...) >= 0.05
BEND_M = (2, 3, 4, 5)  # allowed bending modes (B1 = shear = q, PA)


def bend_keys(keys):
    """Sorted bending modes m present in a parameter-name set ("b2", "b3", ...)."""
    import re

    return tuple(sorted(int(k[1:]) for k in keys if re.fullmatch(r"b\d+", k)))


def bending_of(values, component):
    """Bending of a Sersic from its unpacked values (None without bending modes)."""
    ms = component.bends
    if not ms:
        return None
    return Bending(ms, [values[f"b{m}"] for m in ms], component.bend_pa_ref)


def galfit_bend_sign(pa_deg, pa_ref, m):
    """Factor converting forwardgal's b_m to GALFIT's convention (pa_ref = 0):
    (s s_G)^(m + 1), s = sign cos(PA - pa_ref), s_G = sign cos(PA)."""
    s = jnp.sign(jnp.cos(jnp.deg2rad(pa_deg - pa_ref)))
    s_g = jnp.sign(jnp.cos(jnp.deg2rad(pa_deg)))
    return (s * s_g) ** (m + 1)
# [M12e paused] SPIRAL_THETA_MAX = 720.0  # degrees: default bound on |theta_out| (two full turns)
# [M12e paused] SPIRAL_KEYS = {"power": ("sp_r_in", "sp_dr", "sp_theta", "sp_alpha"),
               # [M12e paused] "log": ("sp_r_in", "sp_dr", "sp_theta", "sp_r_ws")}


# [M12e paused] def spiral_of(values, kind):
    # [M12e paused] """Spiral of a Sersic from its unpacked values (None if kind is None)."""
    # [M12e paused] if kind is None:
        # [M12e paused] return None
    # [M12e paused] shape = values["sp_alpha"] if kind == "power" else values["sp_r_ws"]
    # [M12e paused] return Spiral(kind, values["sp_r_in"], values["sp_r_in"] + values["sp_dr"], values["sp_theta"], shape)


def mode_keys(m, style="cs"):
    """Parameter names of Fourier mode m: ("c1", "s1") or ("a1", "phi1")."""
    return (f"c{m}", f"s{m}") if style == "cs" else (f"a{m}", f"phi{m}")


def _split_mode_keys(keys):
    """(base keys, {m: style}) of a parameter-name set; raises on malformed modes."""
    import re

    base, found = set(), {}
    for k in keys:
        hit = re.fullmatch(r"(c|s|a|phi)(\d+)", k)
        if hit is None:
            base.add(k)
            continue
        m = int(hit.group(2))
        style = "cs" if hit.group(1) in ("c", "s") else "aphi"
        if found.setdefault(m, style) != style:
            raise ValueError(f"mode {m} mixes (c, s) and (a, phi) parameters")
    for m, style in found.items():
        if m not in FOURIER_M:
            raise ValueError(f"Fourier mode m = {m} is not allowed (allowed: {FOURIER_M})")
        if not set(mode_keys(m, style)) <= set(keys):
            raise ValueError(f"mode {m} needs both {mode_keys(m, style)}")
    if len(set(found.values())) > 1:
        raise ValueError("all Fourier modes of a component must use the same style")
    return base, found


def modes_cs(values, ms):
    """Fourier modes of a Sersic from its unpacked values, as CartesianModes
    (None if ms is empty). Works for both parameter styles."""
    if not ms:
        return None
    if f"c{ms[0]}" in values:
        return CartesianModes(tuple(ms), [values[f"c{m}"] for m in ms], [values[f"s{m}"] for m in ms])
    a = [values[f"a{m}"] for m in ms]
    ph = [jnp.deg2rad(values[f"phi{m}"]) for m in ms]
    return CartesianModes(tuple(ms), [ak * jnp.cos(m * pk) for ak, pk, m in zip(a, ph, ms)],
                          [ak * jnp.sin(m * pk) for ak, pk, m in zip(a, ph, ms)])


def aphi_from_cs(c, s, m):
    """GALFIT (a_m >= 0, phi_m in [0, 360/m) deg) from (c_m, s_m); safe at a = 0
    (phase returned as 0 there, where it has no effect)."""
    r2 = c**2 + s**2
    zero = r2 < 1e-24
    a = jnp.sqrt(jnp.where(zero, 1.0, r2)) * jnp.where(zero, 0.0, 1.0)
    phi = jnp.degrees(jnp.arctan2(jnp.where(zero, 0.0, s), jnp.where(zero, 1.0, c))) / m
    return a, jnp.mod(phi, 360.0 / m)


def e_from_qpa(q, pa_deg):
    """(e1, e2) from axis ratio q and PA (deg, east of north)."""
    eps = (1.0 - q) / (1.0 + q)
    phi = np.radians(pa_deg)
    return eps * np.cos(2 * phi), eps * np.sin(2 * phi)


def qpa_from_e(e1, e2):
    """(q, pa_deg) from (e1, e2); safe (finite gradients) at e = 0 (round).

    At e = 0 the PA is undefined; it is returned as 0 there, and the gradient
    w.r.t. PA does not matter because a round profile does not depend on PA.
    PA is in (-90, 90]: add 180 to negatives for reporting in [0, 180).
    """
    eps2 = e1**2 + e2**2
    round_ = eps2 < 1e-24
    eps = jnp.sqrt(jnp.where(round_, 1.0, eps2)) * jnp.where(round_, 0.0, 1.0)
    q = (1.0 - eps) / (1.0 + eps)
    pa = 0.5 * jnp.degrees(jnp.arctan2(jnp.where(round_, 0.0, e2), jnp.where(round_, 1.0, e1)))
    return q, pa


def sersic_qpa(values):
    """(q, pa_deg) of a Sersic from its unpacked values, in either parametrization."""
    if "q" in values:
        return values["q"], values["pa_deg"]
    return qpa_from_e(values["e1"], values["e2"])


@dataclass(frozen=True)
class Component:
    """One model component (see module docstring)."""

    name: str
    kind: str
    plane: str
    params: dict = field(default_factory=dict)
    constraints: tuple = ()  # joint constraints that come with the component (e.g. |e| bound)
    bend_pa_ref: float = 0.0  # PA (deg) defining the bending frame; seam at bend_pa_ref +- 90
    max_snap: float | None = 0.1  # point only: largest snap (px) onto the PSF stamp's own pixel grid
    #   (quasar.snap_to_psf); None = no limit ("trust the PSF")
    psf: dict | None = None  # point only: its own PSF {file, ext, reference}; None = the global psf section
    # [M12e paused] spiral_kind: str = None  # None, "power" or "log" (Sersic spiral rotation)

    def __post_init__(self):
        if self.kind not in SHAPE_PARAMS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.plane not in PLANES:
            raise ValueError(f"{self.name}: plane must be 'source' or 'image', got {self.plane!r}")
        if self.kind == "sky" and self.plane != "image":
            raise ValueError(f"{self.name}: the sky is an image-plane component")
        base, found = _split_mode_keys(self.params) if self.kind == "sersic" else (set(self.params), {})
        bends = bend_keys(base) if self.kind == "sersic" else ()
        bad = [m for m in bends if m not in BEND_M]
        if bad:
            raise ValueError(f"{self.name}: bending mode(s) {bad} not allowed (allowed: {BEND_M})")
        base = base - {f"b{m}" for m in bends}
        fluxes = base & set(FLUX_KEYS)
        if len(fluxes) > 1:
            raise ValueError(f"{self.name}: give its flux as 'flux' or as 'mag', not both")
        base = base - fluxes
        # [M12e paused] if self.spiral_kind is not None:
            # [M12e paused] if self.kind != "sersic" or self.spiral_kind not in SPIRAL_KEYS:
                # [M12e paused] raise ValueError(f"{self.name}: spiral_kind must be 'power' or 'log' on a Sersic")
            # [M12e paused] if not set(SPIRAL_KEYS[self.spiral_kind]) <= base:
                # [M12e paused] raise ValueError(f"{self.name}: a {self.spiral_kind} spiral needs {SPIRAL_KEYS[self.spiral_kind]}")
            # [M12e paused] base = base - set(SPIRAL_KEYS[self.spiral_kind])
        if not any(base == set(ok) for ok in SHAPE_PARAMS[self.kind]):
            options = " or ".join(str(list(ok)) for ok in SHAPE_PARAMS[self.kind])
            raise ValueError(f"{self.name}: {self.kind} needs parameters {options}, got {sorted(self.params)}")
        if "." in self.name:
            raise ValueError(f"component name {self.name!r} must not contain '.'")

    @property
    def flux_key(self):
        """'flux' or 'mag' (the component's flux parameter), or None if it has none."""
        return next((k for k in FLUX_KEYS if k in self.params), None)

    @property
    def names(self):
        """Parameter names in canonical order (Fourier modes, bending, then the flux last)."""
        base, found = _split_mode_keys(self.params) if self.kind == "sersic" else (set(self.params), {})
        # [M12e paused] spiral = SPIRAL_KEYS[self.spiral_kind] if self.spiral_kind is not None else ()
        # [M12e paused] names = next(ok for ok in SHAPE_PARAMS[self.kind] if base - set(spiral) == set(ok))
        # [M12e paused] return names + tuple(k for m in sorted(found) for k in mode_keys(m, found[m])) + spiral
        bends = bend_keys(base) if self.kind == "sersic" else ()
        base = base - {f"b{m}" for m in bends} - set(FLUX_KEYS)
        names = next(ok for ok in SHAPE_PARAMS[self.kind] if base == set(ok))  # [M12e stand-in]
        flux = (self.flux_key,) if self.flux_key else ()
        return (names + tuple(k for m in sorted(found) for k in mode_keys(m, found[m]))  # [M12e stand-in]
                + tuple(f"b{m}" for m in bends) + flux)

    @property
    def bends(self):
        """Bending modes present, sorted: e.g. (2, 3); () if none."""
        return bend_keys(self.params) if self.kind == "sersic" else ()

    @property
    def modes(self):
        """Fourier modes present, sorted: e.g. (1, 4); () if none."""
        return tuple(sorted(_split_mode_keys(self.params)[1])) if self.kind == "sersic" else ()


def sersic_component(name, plane, x0, y0, R_e, n, q, pa_deg, overrides=None, ellipticity="e1e2",
                     # [M12e paused] modes=None, mode_style="cs", spiral=None):
                     modes=None, mode_style="cs", bending=None):  # [M12e stand-in]
    """A Sersic component with the default constraints (user's choice, 2026-11-16).

    Defaults: x0, y0 free, unbounded; R_e >= 0.01" (stops a component from
    collapsing into a point source, where its unit-flux normalization
    overflows), and R_e < 2" in the source plane (2026-11-22: stops a huge
    faint disk from absorbing the sky and quasar light); 0.5 < n < 8 (the
    range the renderer is verified for); q > 0.05.

    ellipticity "e1e2" (default): shape parameters (e1, e2), started from the
    given q and pa_deg, with the joint bound |e| < EPS_MAX (q > 0.05) attached
    to the component (see module docstring). "qpa": parameters q (0.05 < q < 1)
    and pa_deg (0 <= pa_deg < 180; a PA and PA + 180 are the same orientation;
    the starting PA is wrapped into this range; a true PA near 0 / 180 deg then
    sits at a bound). ``overrides``: dict mapping a parameter name to a
    ``Param`` that replaces the default, e.g. {"n": Param(4.0, free=False)}.

    modes : None, iterable of m, or dict {m: (a_m, phi_m_deg)}
        Fourier modes (m in 1, 3, 4) with GALFIT starting values (a list of m
        starts them at a_m = 0). mode_style "cs" (default): parameters
        (c_m, s_m) with the joint bound sum_m a_m < FOURIER_SUM_MAX
        (``FourierBound``). "aphi": parameters (a_m, phi_m) directly,
        |a_m| < FOURIER_SUM_MAX / K, phi_m unbounded (periodic).
    bending : None, iterable of m, or dict {m: b_m}
        Bending modes (m in 2..5) with starting amplitudes in arcsec (a list
        starts them at 0). Parameters b{m}, unbounded; the bend frame's seam
        lies 90 deg from the starting PA (see the module docstring).
    """
    r_hi = R_E_MAX_SOURCE if plane == "source" else None
    params = {"x0": Param(x0), "y0": Param(y0), "R_e": Param(R_e, lo=R_E_MIN, hi=r_hi), "n": Param(n, lo=0.5, hi=8.0)}
    constraints = ()
    if ellipticity == "e1e2":
        e1, e2 = e_from_qpa(q, pa_deg)
        params["e1"], params["e2"] = Param(float(e1)), Param(float(e2))
    elif ellipticity == "qpa":
        params["q"] = Param(q, lo=Q_MIN, hi=1.0)
        # lower bound a hair below 0 so that 0 deg itself is a valid value
        params["pa_deg"] = Param(float(pa_deg) % 180.0, lo=-1e-6, hi=180.0)
    else:
        raise ValueError(f"ellipticity must be 'e1e2' or 'qpa', got {ellipticity!r}")
    start_modes = {} if modes is None else (dict(modes) if isinstance(modes, dict)
                                             else {m: (0.0, 0.0) for m in modes})
    for m in start_modes:
        if m not in FOURIER_M:
            raise ValueError(f"Fourier mode m = {m} is not allowed (allowed: {FOURIER_M})")
    if sum(abs(a) for a, _ in start_modes.values()) >= FOURIER_SUM_MAX:
        raise ValueError(f"{name}: starting mode amplitudes must sum to < {FOURIER_SUM_MAX}")
    ms = tuple(sorted(start_modes))
    for m in ms:
        a, ph = start_modes[m]
        if mode_style == "cs":
            params[f"c{m}"] = Param(float(a * np.cos(m * np.radians(ph))))
            params[f"s{m}"] = Param(float(a * np.sin(m * np.radians(ph))))
        elif mode_style == "aphi":
            lim = FOURIER_SUM_MAX / len(ms)
            if abs(a) >= lim:
                raise ValueError(f"{name}: |a_{m}| must start below {lim:.3f} with mode_style='aphi'")
            params[f"a{m}"] = Param(float(a), lo=-lim, hi=lim)
            params[f"phi{m}"] = Param(float(ph))
        else:
            raise ValueError(f"mode_style must be 'cs' or 'aphi', got {mode_style!r}")
    start_bends = {} if bending is None else (dict(bending) if isinstance(bending, dict)
                                               else {m: 0.0 for m in bending})
    for m in start_bends:
        if m not in BEND_M:
            raise ValueError(f"bending mode m = {m} is not allowed (allowed: {BEND_M})")
    for m in sorted(start_bends):
        params[f"b{m}"] = Param(float(start_bends[m]))
    # [M12e paused] spiral_kind = None
    # [M12e paused] if spiral is not None:
        # [M12e paused] sp = dict(spiral)
        # [M12e paused] spiral_kind = sp.pop("kind")
        # [M12e paused] if spiral_kind not in SPIRAL_KEYS:
            # [M12e paused] raise ValueError(f"spiral kind must be 'power' or 'log', got {spiral_kind!r}")
        # [M12e paused] if not 0 <= sp["r_in"] < sp["r_out"]:
            # [M12e paused] raise ValueError(f"{name}: spiral needs 0 <= r_in < r_out")
        # [M12e paused] params["sp_r_in"] = Param(float(sp["r_in"]), lo=0.0)
        # [M12e paused] if sp["r_out"] - sp["r_in"] <= R_E_MIN or abs(sp["theta_out"]) >= SPIRAL_THETA_MAX:
            # [M12e paused] raise ValueError(f"{name}: spiral needs r_out - r_in > {R_E_MIN} and |theta_out| < {SPIRAL_THETA_MAX}")
        # [M12e paused] params["sp_dr"] = Param(float(sp["r_out"] - sp["r_in"]), lo=R_E_MIN)
        # [M12e paused] params["sp_theta"] = Param(float(sp["theta_out"]), lo=-SPIRAL_THETA_MAX, hi=SPIRAL_THETA_MAX)
        # [M12e paused] if spiral_kind == "power":
            # [M12e paused] params["sp_alpha"] = Param(float(sp.get("alpha", 0.0)), lo=-2.0, hi=3.0)
        # [M12e paused] else:
            # [M12e paused] params["sp_r_ws"] = Param(float(sp["r_ws"]), lo=0.0)
    params.update(overrides or {})
    if ms and mode_style == "cs":
        keys = [k for m in ms for k in mode_keys(m)]
        if all(params[k].free and params[k].lo is None and params[k].hi is None for k in keys):
            constraints += (FourierBound(name, ms),)
        else:
            warnings.warn(f"{name}: a Fourier (c, s) parameter is fixed or bounded; the joint bound "
                          f"sum a_m < {FOURIER_SUM_MAX} is NOT applied (g may become <= 0)", stacklevel=2)
    if ellipticity == "e1e2" and all(params[k].free and params[k].lo is None and params[k].hi is None
                                     for k in ("e1", "e2")):
        constraints += (PositionConstraint(name, (0.0, 0.0), EPS_MAX, np.eye(2), "bounds",
                                           f"|e| < {EPS_MAX:.3f} (q > {Q_MIN:g}) [hard bound]", ("e1", "e2")),)
    # Any parameter may be unbounded, but warn when that allows invalid values.
    r, nn = params["R_e"], params["n"]
    if r.free and (r.lo is None or r.lo < 0):
        warnings.warn(f"{name}: R_e may become <= 0 (no lower bound at 0)", stacklevel=2)
    if nn.free and (nn.lo is None or nn.lo < 0.5 or nn.hi is None or nn.hi > 8.0):
        warnings.warn(f"{name}: n may leave 0.5-8, the range the exact pixel integration is verified for",
                      stacklevel=2)
    # [M12e paused] return Component(name=name, kind="sersic", plane=plane, params=params, constraints=constraints,
                     # [M12e paused] spiral_kind=spiral_kind)
    return Component(name=name, kind="sersic", plane=plane, params=params, constraints=constraints,  # [M12e stand-in]
                     bend_pa_ref=float(pa_deg))


def regularize(component, n_sigma=2.0, re_frac=1.0, e_sigma=0.4):
    """Opt-in weak Gaussian priors on a Sersic's shape (a "ridge" for shapes).

    n ~ N(n_start, n_sigma), R_e ~ N(R_e_start, re_frac * R_e_start), and, for
    (e1, e2), each ~ N(0, e_sigma) (mildly favours rounder shapes). They keep
    degenerate directions from drifting to the bounds; the price is a small
    bias towards these values. Returns a new Component.
    """
    if component.kind != "sersic":
        raise ValueError("regularize applies to Sersic components")
    p = dict(component.params)

    def with_prior(prm, mu, sigma):
        return Param(prm.value, free=prm.free, lo=prm.lo, hi=prm.hi, prior_mu=mu, prior_sigma=sigma)

    p["n"] = with_prior(p["n"], p["n"].value, n_sigma)
    p["R_e"] = with_prior(p["R_e"], p["R_e"].value, re_frac * p["R_e"].value)
    if "e1" in p:
        p["e1"] = with_prior(p["e1"], 0.0, e_sigma)
        p["e2"] = with_prior(p["e2"], 0.0, e_sigma)
    # [M12e paused] return Component(component.name, component.kind, component.plane, p, component.constraints,
                     # [M12e paused] component.spiral_kind)
    return replace(component, params=p)  # [M12e stand-in]: every other field (options) carried over


def point_component(name, plane, x0, y0, overrides=None, max_snap=0.1, psf=None):
    """A point source (quasar: plane "source"; a foreground star: plane "image").

    max_snap : largest snap (px) of a locked image position onto the PSF
        stamp's pixel grid (see quasar.snap_to_psf); None = no limit.
    psf : its own PSF, {"file": ..., "ext": 0, "reference": "centroid"};
        None = the configuration's global psf section.
    """
    params = {"x0": Param(x0), "y0": Param(y0)}
    params.update(overrides or {})
    return Component(name=name, kind="point", plane=plane, params=params, max_snap=max_snap, psf=psf)


def sky_component(name="sky"):
    return Component(name=name, kind="sky", plane="image", params={})


def pa_window(pa_deg, free=True):
    """PA bounded to (pa - 90, pa + 90): removes the PA <-> PA + 180 duplicate.

    Only applied when the user asks for it: overrides={"pa_deg": pa_window(pa)}.
    """
    return Param(pa_deg, free=free, lo=pa_deg - 90.0, hi=pa_deg + 90.0)


class ModelParams:
    """All components' parameters as one flat unconstrained vector.

    Parameters
    ----------
    components : list of Component
    ties : dict, optional
        {"Comp.param": ("Other.param", offset)}: the first always equals the
        second plus offset. Both must be in the same plane.
    constraints : list of PositionConstraint, optional
        Joint 2-D constraints on parameter pairs (e.g. "centre within N",
        ``within``), in addition to those that come with the components (the
        |e| bound). At most one per (component, pair). The constrained
        parameters must be free, unbounded and not tied.
    """

    def __init__(self, components, ties=None, constraints=None):
        self.components = list(components)
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"component names must be unique: {names}")
        self.by_name = {c.name: c for c in self.components}
        self.ties = dict(ties or {})
        for target, (source, _) in self.ties.items():
            for key in (target, source):
                self._lookup(key)
            if self.by_name[target.split(".")[0]].plane != self.by_name[source.split(".")[0]].plane:
                raise ValueError(f"tie {target} <- {source} mixes source- and image-plane coordinates")
            if source in self.ties:
                raise ValueError(f"tie chains are not supported: {source} is itself tied")
            if target == source:
                raise ValueError(f"{target} cannot be tied to itself")
        self.constraints = {}  # (component, keys) -> PositionConstraint
        self.fourier_bounds = {}  # component -> FourierBound
        all_constraints = [pc for c in self.components for pc in c.constraints] + list(constraints or [])
        for fb in [pc for pc in all_constraints if isinstance(pc, FourierBound)]:
            if fb.component in self.fourier_bounds:
                raise ValueError(f"{fb.component}: only one Fourier bound")
            for par in fb.keys:
                key = f"{fb.component}.{par}"
                prm = self._lookup(key)
                if not prm.free or prm.lo is not None or prm.hi is not None or key in self.ties:
                    raise ValueError(f"{key} must be free, unbounded and untied to carry the Fourier bound")
            self.fourier_bounds[fb.component] = fb
        for pc in [pc for pc in all_constraints if not isinstance(pc, FourierBound)]:
            if pc.component not in self.by_name:
                raise ValueError(f"constraint for unknown component {pc.component!r}")
            key2 = (pc.component, tuple(pc.keys))
            if key2 in self.constraints:
                raise ValueError(f"{pc.component}: only one constraint on {pc.keys}")
            for par in pc.keys:
                key = f"{pc.component}.{par}"
                prm = self._lookup(key)
                if not prm.free or prm.lo is not None or prm.hi is not None or key in self.ties:
                    raise ValueError(f"{key} must be free, unbounded and untied to carry a joint constraint")
                if pc.keys == ("x0", "y0") and prm.prior_mu is not None:
                    raise ValueError(f"{key} must not have its own prior to carry a position constraint")
            self.constraints[key2] = pc
        # The free vector: free, untied parameters in component order.
        self.free_keys = [
            f"{c.name}.{p}" for c in self.components for p in c.names
            if c.params[p].free and f"{c.name}.{p}" not in self.ties
        ]

    def _lookup(self, key):
        comp, _, par = key.partition(".")
        if comp not in self.by_name or par not in self.by_name[comp].params:
            raise ValueError(f"unknown parameter {key!r}")
        return self.by_name[comp].params[par]

    @property
    def size(self):
        return len(self.free_keys)

    def _pair_index(self, pc):
        return tuple(self.free_keys.index(f"{pc.component}.{k}") for k in pc.keys)

    def start(self):
        """u at the starting values."""
        u = [float(self._lookup(k).to_unconstrained(self._lookup(k).value)) for k in self.free_keys]
        for pc in self.constraints.values():
            if pc.mode != "bounds":
                continue
            c = self.by_name[pc.component]
            a, b = pc.keys
            w1, w2 = pc.w_of_offset(c.params[a].value - pc.ref[0], c.params[b].value - pc.ref[1])
            r2 = w1**2 + w2**2
            if r2 >= 1.0:
                raise ValueError(f"{pc.component}: starting {pc.keys} lie outside the constraint ({pc.description})")
            i, j = self._pair_index(pc)
            u[i], u[j] = w1 / np.sqrt(1.0 - r2), w2 / np.sqrt(1.0 - r2)  # inverse of w = u / sqrt(1 + |u|^2)
        for fb in self.fourier_bounds.values():
            c = self.by_name[fb.component]
            w = fb.to_unconstrained(np.array([c.params[k].value for k in fb.keys]))
            for k, wk in zip(fb.keys, w):
                u[self.free_keys.index(f"{fb.component}.{k}")] = float(wk)
        return jnp.array(u)

    def _fourier_index(self, fb):
        return [self.free_keys.index(f"{fb.component}.{k}") for k in fb.keys]

    def unpack(self, u):
        """u -> {component name: {param: physical value}} (fixed and tied included)."""
        values = {c.name: {p: jnp.asarray(c.params[p].value, dtype=float) for p in c.names}
                  for c in self.components}
        for i, key in enumerate(self.free_keys):
            comp, _, par = key.partition(".")
            values[comp][par] = self._lookup(key).to_physical(u[i])
        for pc in self.constraints.values():
            if pc.mode == "bounds":  # joint 2-D map of the plane onto the allowed disk / ellipse
                i, j = self._pair_index(pc)
                scale = 1.0 / jnp.sqrt(1.0 + u[i] ** 2 + u[j] ** 2)
                dx, dy = pc.offset_of_w(u[i] * scale, u[j] * scale)
                a, b = pc.keys
                values[pc.component][a] = pc.ref[0] + dx
                values[pc.component][b] = pc.ref[1] + dy
        for fb in self.fourier_bounds.values():
            y = fb.to_physical(jnp.stack([u[i] for i in self._fourier_index(fb)]))
            for k, yk in zip(fb.keys, y):
                values[fb.component][k] = yk
        for target, (source, offset) in self.ties.items():
            sc, _, sp = source.partition(".")
            tc, _, tp = target.partition(".")
            values[tc][tp] = values[sc][sp] + offset
        return values

    def log_prior(self, u, jacobian=True):
        """Sum of Gaussian log priors of all parameters (+ log|dx/du| if jacobian).

        Use jacobian=True for sampling in u (NUTS), False for optimization.
        Priors on tied parameters are evaluated at their tied values.
        """
        values = self.unpack(u)
        total = 0.0
        for c in self.components:
            for p in c.names:
                total = total + c.params[p].log_prior(values[c.name][p])
        for pc in self.constraints.values():
            if pc.mode == "gaussian":
                a, b = pc.keys
                w1, w2 = pc.w_of_offset(values[pc.component][a] - pc.ref[0], values[pc.component][b] - pc.ref[1])
                total = total - 0.5 * (w1**2 + w2**2)
        if jacobian:
            joint = {f"{pc.component}.{k}" for pc in self.constraints.values() if pc.mode == "bounds" for k in pc.keys}
            joint |= {f"{fb.component}.{k}" for fb in self.fourier_bounds.values() for k in fb.keys}
            for fb in self.fourier_bounds.values():
                total = total + fb.log_jacobian(jnp.stack([u[i] for i in self._fourier_index(fb)]))
            for i, key in enumerate(self.free_keys):
                if key not in joint:
                    total = total + self._lookup(key).log_jacobian(u[i])
            for pc in self.constraints.values():
                if pc.mode == "bounds":
                    i, j = self._pair_index(pc)
                    total = (total + np.log(pc.radius**2 * abs(np.linalg.det(pc.matrix)))
                             - 2.0 * jnp.log(1.0 + u[i] ** 2 + u[j] ** 2))
        return total

    def prior_residuals(self, u):
        """Gaussian priors as residuals: sum of squares = -2 log_prior(u, jacobian=False).

        One entry (x - mu) / sigma per parameter with a Gaussian prior (at its
        tied value if tied), and (w1, w2) per Gaussian "within" constraint.
        Used by least-squares optimizers (Levenberg-Marquardt).
        """
        values = self.unpack(u)
        res = []
        for c in self.components:
            for p in c.names:
                prm = c.params[p]
                if prm.prior_mu is not None:
                    res.append((values[c.name][p] - prm.prior_mu) / prm.prior_sigma)
        for pc in self.constraints.values():
            if pc.mode == "gaussian":
                a, b = pc.keys
                w1, w2 = pc.w_of_offset(values[pc.component][a] - pc.ref[0], values[pc.component][b] - pc.ref[1])
                res += [w1, w2]
        return jnp.stack([jnp.asarray(r, dtype=float) for r in res]) if res else jnp.zeros(0)

    def describe(self):
        lines = [f"{self.size} free parameters"]
        for c in self.components:
            lines.append(f"{c.name}: {c.kind}, {c.plane} plane")
            for p in c.names:
                key = f"{c.name}.{p}"
                if key in self.ties:
                    src, off = self.ties[key]
                    lines.append(f"    {p:7s} tied to {src}" + (f" + {off:g}" if off else ""))
                else:
                    lines.append(f"    {p:7s} {c.params[p].describe()}")
            if "e1" in c.params:
                q0, pa0 = qpa_from_e(c.params["e1"].value, c.params["e2"].value)
                lines.append(f"    (start q = {float(q0):.3f}, PA = {float(pa0) % 180:.1f} deg)")
            for pc in self.constraints.values():
                if pc.component == c.name:
                    lines.append(f"    {'centre' if pc.keys == ('x0', 'y0') else ','.join(pc.keys):7s} {pc.description}")
            if c.name in self.fourier_bounds:
                lines.append(f"    {'modes':7s} {self.fourier_bounds[c.name].description}")
        return "\n".join(lines)


# --- "centre within N" constraints -------------------------------------------------


@dataclass(frozen=True)
class PositionConstraint:
    """Keep a parameter PAIR (default the centre x0, y0; or e1, e2) near a reference.

    Delta = N M w, |w| < 1.

    Attributes
    ----------
    component : str
        Component name.
    ref : (float, float)
        Reference centre in the component's own plane (arcsec).
    radius : float
        N (in the chosen unit).
    matrix : 2 x 2 array
        M: maps the dimensionless offset w to arcsec in the component's plane
        (see ``within``).
    mode : "bounds" or "gaussian"
        "bounds": (x0, y0) is kept strictly inside |w| < 1 by the smooth map
        w = u / sqrt(1 + |u|^2). "gaussian": prior -|w|^2 / 2 (1 sigma = N).
    description : str
        Human-readable form, e.g. "within 2 px (image plane) of (-5.73, 0.19)".
    """

    component: str
    ref: tuple
    radius: float
    matrix: np.ndarray
    mode: str
    description: str
    keys: tuple = ("x0", "y0")  # the parameter pair it constrains

    def w_of_offset(self, dx, dy):
        Minv = np.linalg.inv(self.matrix)
        return (Minv[0, 0] * dx + Minv[0, 1] * dy) / self.radius, (Minv[1, 0] * dx + Minv[1, 1] * dy) / self.radius

    def offset_of_w(self, w1, w2):
        M = self.matrix
        return self.radius * (M[0, 0] * w1 + M[0, 1] * w2), self.radius * (M[1, 0] * w1 + M[1, 1] * w2)


@dataclass(frozen=True)
class FourierBound:
    """Joint bound sum_m a_m < limit on a Sersic's Cartesian Fourier modes.

    With y_m = (c_m, s_m) (so a_m = |y_m|) and unconstrained 2-vectors w_m:

        y_m = limit * w_m / D,   D = 1 + sum_k rho(w_k),
        rho(w) = sqrt(|w|^2 + delta^2) - delta     (smooth version of |w|).

    Since |w| <= rho(w) + delta, sum_m a_m <= limit (sum rho + K delta) /
    (1 + sum rho) < limit whenever K delta < 1. One mode alone can approach
    the limit. The map is smooth everywhere (also at w = 0, the usual start).
    """

    component: str
    ms: tuple
    limit: float = FOURIER_SUM_MAX
    delta: float = 0.01

    def __post_init__(self):
        if len(self.ms) * self.delta >= 1.0:
            raise ValueError("FourierBound needs K * delta < 1")

    @property
    def keys(self):
        return tuple(k for m in self.ms for k in mode_keys(m))

    @property
    def description(self):
        return f"sum a_m < {self.limit:g} over m = {self.ms} [hard bound]"

    def _rho(self, w):  # w: (K, 2)
        return jnp.sqrt(jnp.sum(w**2, axis=1) + self.delta**2) - self.delta

    def to_physical(self, w_flat):
        """(2K,) unconstrained -> (2K,) (c_1, s_1, c_3, s_3, ...)."""
        w = jnp.reshape(w_flat, (-1, 2))
        return jnp.ravel(self.limit * w / (1.0 + jnp.sum(self._rho(w))))

    def to_unconstrained(self, y_flat):
        """Inverse (numpy): w_m = kappa y_m with one kappa > 0 solving
        limit kappa = 1 + sum_k rho(kappa y_k) (increasing in kappa; bisection)."""
        y = np.reshape(np.asarray(y_flat, dtype=float), (-1, 2))
        r = np.hypot(y[:, 0], y[:, 1])
        if r.sum() >= self.limit:
            raise ValueError(f"{self.component}: starting mode amplitudes sum to >= {self.limit}")
        if not np.any(r > 0):
            return np.zeros(y.size)

        def f(k):
            return self.limit * k - 1.0 - np.sum(np.sqrt((k * r) ** 2 + self.delta**2) - self.delta)

        lo, hi = 0.0, 1.0
        while f(hi) < 0:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return np.ravel(0.5 * (lo + hi) * y)

    def log_jacobian(self, w_flat):
        """log |det d y / d w| (matrix determinant lemma): for NUTS."""
        w = jnp.reshape(w_flat, (-1, 2))
        D = 1.0 + jnp.sum(self._rho(w))
        wgrad = jnp.sum(jnp.sum(w**2, axis=1) / jnp.sqrt(jnp.sum(w**2, axis=1) + self.delta**2))
        return w.size * jnp.log(self.limit / D) + jnp.log(1.0 - wgrad / D)


def lens_jacobian(field, theta):
    """A = d beta / d theta (2 x 2) at an image-plane position theta (arcsec)."""
    import jax

    f = lambda t: jnp.stack(field.backtrace(t[0], t[1]))  # noqa: E731
    return np.asarray(jax.jacfwd(f)(jnp.asarray(theta, dtype=float)))


def within(component, ref, radius, unit, mode="bounds", pixscale=None, lens_matrix=None):
    """Constraint: the centre of ``component`` within ``radius`` of ``ref``.

    unit "arcsec": radius in arcsec in the component's own plane (beta for a
    source-plane component, theta for an image-plane one).
    unit "px": radius in image-plane data pixels. Image-plane component:
    M = pixscale I. Source-plane component: its LENSED centre may move at most
    radius px, using the local lens mapping at the reference: d beta = A d theta,
    so M = A pixscale, with A = ``lens_jacobian`` at the reference centre's
    image (pass it as ``lens_matrix``). The allowed region in beta is then an
    ellipse.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if mode not in ("bounds", "gaussian"):
        raise ValueError(f"mode must be 'bounds' or 'gaussian', got {mode!r}")
    if unit == "arcsec":
        M = np.eye(2)
        desc = f"within {radius:g}\" ({component.plane} plane) of ({ref[0]:+.4f}, {ref[1]:+.4f})"
    elif unit == "px":
        if pixscale is None:
            raise ValueError("unit 'px' needs pixscale")
        if component.plane == "image":
            M = pixscale * np.eye(2)
        else:
            if lens_matrix is None:
                raise ValueError("unit 'px' for a source-plane component needs lens_matrix (lens_jacobian)")
            M = np.asarray(lens_matrix, float) * pixscale
        desc = f"within {radius:g} px (image plane) of ({ref[0]:+.4f}, {ref[1]:+.4f})"
    else:
        raise ValueError(f"unit must be 'arcsec' or 'px', got {unit!r}")
    desc += " [hard bound]" if mode == "bounds" else " [Gaussian, 1 sigma]"
    return PositionConstraint(component.name, tuple(map(float, ref)), float(radius), M, mode, desc)
