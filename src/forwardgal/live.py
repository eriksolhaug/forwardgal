"""Fast, repeated rendering of a model spec for the GUI (G1a).

The GUI edits the model spec (the ``model:`` section of a model YAML, as a
dict: forwardgal.serialize) and asks for the model image after every edit.
``LiveModel`` answers quickly by

  - rendering each component with ``model.component_template``, the same
    function the fitter uses, jitted once per component *structure* (kind,
    plane, parameter names, Fourier/bending modes) with the parameter values
    as inputs, so new values never trigger a recompile;
  - caching each component's template together with the values it was made
    from: dragging one component re-renders only that one, because the model
    is linear in the components, M = sum_k a_k T_k;
  - taking each component's amplitude from its flux parameter ('flux' or
    'mag', F4a, 2026-11-22): dragging never changes a flux; components
    without one (old files) are solved by the fitter's NNLS on the data
    minus the others. ``best_fluxes`` is the "Fit fluxes" button: one NNLS
    solve for the free fluxes, on request only.

Arrays are on the render box of the Setup ([row, col], 0-based, shape
``grid.shape``). Amplitude priors of the fitter are not applied here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from forwardgal.components import FLUX_KEYS, Q_MIN, bending_of, e_from_qpa, modes_cs, sersic_qpa
from forwardgal.likelihood import _solve_numpy
from forwardgal.model import component_template
from forwardgal.photometry import mag_to_flux, sb_to_sky
from forwardgal.profiles.sersic import sersic, unit_flux_ie
from forwardgal.serialize import model_from_dict


@dataclass
class LiveResult:
    """One rendering of the spec on the render box ((ny, nx) arrays)."""

    model: np.ndarray  # sum_k a_k T_k (data units)
    residual: np.ndarray  # (data - model) / sigma; NaN outside the mask
    chi2: float  # in the mask
    dof: int  # mask pixels - amplitudes - free shape parameters
    amplitudes: dict  # name -> a_k (total flux in data units; sky: per pixel)


def structure_key(c):
    """What a component's compiled template function depends on (not its values)."""
    return (c.kind, c.plane, tuple(sorted(c.params)), c.modes, c.bends, float(c.bend_pa_ref))


class LiveModel:
    """Render a model spec on a Setup's render box, caching per component.

    setup : forwardgal.config.Setup (box, traced grid, PSF, data, mask, quasar)
    spec : model dict (default: the setup's own cfg["model"])
    """

    def __init__(self, setup, spec=None):
        m = setup.model
        self.grid, self.convolve, self.pixels = m.grid, m.convolve, m.pixels
        self.pixscale, self.settings, self.method = m.pixscale, m.settings, m.method
        self.zeropoint = m.zeropoint
        self.data, self.sigma = np.asarray(m.data, float), np.asarray(m.sigma, float)
        self.mask = np.asarray(m.mask, bool)
        self.fixed = {k: np.asarray(t, float) for k, t in m.fixed_templates.items()}
        # Point components (locked image positions): their templates come from config.build.
        self.points = {k: np.asarray(t, float) for k, t in m.point_templates.items()}
        self._point_at = {n: (m.params.by_name[n].params["x0"].value, m.params.by_name[n].params["y0"].value)
                          for n in self.points}
        self._compiled = {}  # (purpose, structure_key) -> jitted function
        self._cache = {}  # component name -> ((structure, values), template)
        self.off = set()  # components left out of the model for now (GUI "in model" box; the spec keeps them)
        self.set_spec(setup.cfg["model"] if spec is None else spec)

    # --- the spec ---------------------------------------------------------------

    def set_spec(self, spec):
        """Take a new model spec. Cached templates of unchanged components are kept."""
        params = model_from_dict(spec)
        values = params.unpack(params.start())  # ties and fixed values resolved
        self.params = params
        self.components = {c.name: c for c in params.components}
        self.values = {c.name: {k: float(x) for k, x in values[c.name].items()} for c in params.components}
        self._cache = {k: v for k, v in self._cache.items() if k in self.components}
        self.off &= set(self.components)  # (a removed component is no longer "off")

    # --- rendering ----------------------------------------------------------------

    def _jit(self, purpose, c, fn):
        key = (purpose, structure_key(c))
        if key not in self._compiled:
            self._compiled[key] = jax.jit(fn)
        return self._compiled[key]

    def template(self, name):
        """PSF-convolved unit-flux template of component ``name`` (cached)."""
        c, v = self.components[name], self.values[name]
        if c.kind == "point":  # made once by config.build, for the position it was built at
            if name not in self.points or (v["x0"], v["y0"]) != self._point_at[name]:
                raise NotImplementedError(f"{name}: a point's position is locked; moving or adding one "
                                          "needs the model to be built again")
            return self.points[name]
        shape = {k: x for k, x in v.items() if k not in FLUX_KEYS}  # a template has unit flux
        key = (structure_key(c), tuple(sorted(shape.items())))
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        fn = self._jit("template", c, lambda vv, c=c: component_template(
            c, vv, self.grid, self.convolve, self.pixels, self.pixscale, self.settings))
        t = np.asarray(fn({k: jnp.asarray(x) for k, x in shape.items()}))
        self._cache[name] = (key, t)
        return t

    def amplitude(self, name):
        """Amplitude given by component ``name``'s flux parameter (photometry.py), or None."""
        c = self.components[name]
        if c.flux_key is None:
            return None
        x = self.values[name][c.flux_key]
        if c.flux_key == "flux":
            return x
        if self.zeropoint is None:
            raise ValueError(f"{name}: a 'mag' flux parameter needs the image's zeropoint")
        return sb_to_sky(x, self.zeropoint, self.pixscale) if c.kind == "sky" else mag_to_flux(x, self.zeropoint)

    def _solve(self, names, T, d):
        """NNLS amplitudes (fluxes >= 0, sky free) of templates T (K, ny, nx) for data d in the mask."""
        nonneg = [self.components[n].kind != "sky" if n in self.components else True for n in names]
        nan = np.full(len(names), np.nan)
        return np.asarray(_solve_numpy(T[:, self.mask].T, d[self.mask], self.sigma[self.mask], self.method,
                                       nonneg, 1e-6, nan, nan), float)

    def render(self):
        """LiveResult for the current spec: amplitudes from the flux parameters (others: NNLS)."""
        names = list(self.components) + list(self.fixed)
        T = np.stack([self.template(n) for n in self.components] + list(self.fixed.values()))
        a = np.array([0.0 if n in self.off else
                      self.amplitude(n) if n in self.components and self.components[n].flux_key else np.nan
                      for n in names])  # an "off" component contributes nothing
        solved = np.isnan(a)
        if solved.any():  # components without a flux parameter: NNLS on the data minus the given ones
            rest = self.data - np.tensordot(np.where(solved, 0.0, a), T, axes=1)
            a[solved] = self._solve([n for n, s in zip(names, solved) if s], T[solved], rest)
        model = np.tensordot(a, T, axes=1)
        res = np.where(self.mask, (self.data - model) / self.sigma, np.nan)
        chi2 = float(np.nansum(res**2))
        dof = int(self.mask.sum()) - int(solved.sum()) - self.params.size
        return LiveResult(model, res, chi2, dof, dict(zip(names, a.tolist())))

    def best_fluxes(self, names=None):
        """The "Fit fluxes" button: NNLS amplitudes of the components whose flux may change.

        names : the components to fit (the GUI's selection); None = all.
        Of those, components with a free, untied flux parameter are solved;
        every other component (not in ``names``, or with a locked or tied
        flux) is kept at its flux and subtracted first. Components in
        ``off`` are left out entirely (as if absent), and never solved. Templates without a
        flux parameter are always solved, as in ``render``. Shapes are not
        changed. Returns {name: amplitude} of the solved ones.
        """
        chosen = set(self.components) if names is None else set(names)
        unknown = chosen - set(self.components)
        if unknown:
            raise KeyError(f"no component(s) {sorted(unknown)}")
        everything = list(self.components) + list(self.fixed)
        T = np.stack([self.template(n) for n in self.components] + list(self.fixed.values()))
        free = []
        for n in everything:
            c = self.components.get(n)
            if n in self.off:
                free.append(False)
            elif c is None or c.flux_key is None:  # no flux parameter: solved by every render anyway
                free.append(True)
            elif n not in chosen:
                free.append(False)
            else:
                free.append(c.params[c.flux_key].free and f"{n}.{c.flux_key}" not in self.params.ties)
        free = np.array(free)
        keep = np.array([0.0 if f or n in self.off else self.amplitude(n) for n, f in zip(everything, free)])
        rest = self.data - np.tensordot(keep, T, axes=1)
        solved = [n for n, f in zip(everything, free) if f]
        a = self._solve(solved, T[free], rest)
        return dict(zip(solved, a.tolist()))

    def source_image(self, bx, by, amplitudes):
        """Surface brightness of the source-plane components at beta = (bx, by).

        bx, by : arrays (arcsec), any matching shape (e.g. a display grid)
        amplitudes : LiveResult.amplitudes
        Returns an array like bx in data units per image pixel area
        (pixscale^2), i.e. what an unlensed pixel there would record.
        Point-sampled (no pixel integration): for display only.
        """
        out = np.zeros(np.shape(bx))
        for name, c in self.components.items():
            if c.kind != "sersic" or c.plane != "source":
                continue

            def sb(vv, x, y, c=c):
                q, pa = sersic_qpa(vv)
                modes = modes_cs(vv, c.modes)
                ie = unit_flux_ie(vv["R_e"], vv["n"], q, self.pixscale, modes=modes)
                return sersic(x, y, vv["x0"], vv["y0"], ie, vv["R_e"], vv["n"], q, pa, modes=modes,
                              bending=bending_of(vv, c))

            v = {k: jnp.asarray(x) for k, x in self.values[name].items()}
            out += amplitudes[name] * np.asarray(self._jit("source", c, sb)(v, jnp.asarray(bx), jnp.asarray(by)))
        return out


# --- editing the spec -------------------------------------------------------------


def _into_bounds(x, p):
    """Clip value x into a parameter dict's (lo, hi), strictly inside."""
    lo, hi = p.get("lo"), p.get("hi")
    eps = 1e-9
    if lo is not None and x <= lo:
        x = lo + eps * max(1.0, abs(lo))
    if hi is not None and x >= hi:
        x = hi - eps * max(1.0, abs(hi))
    return x


def set_ellipse(spec, name, x0=None, y0=None, R_e=None, q=None, pa_deg=None):
    """Copy of ``spec`` with component ``name``'s ellipse changed (None: unchanged).

    x0, y0 : centre (arcsec, its own plane; +x east, +y north)
    R_e : semi-major axis (arcsec); q : axis ratio b/a; pa_deg : PA east of north
    A component in (e1, e2) gets e1, e2 from (q, PA). Values are clipped into
    each parameter's bounds (q into [Q_MIN, 1]); a PA with bounds is first
    moved by multiples of 180 deg towards them (an ellipse's PA is mod 180).
    The values of fixed parameters are changed too (they are where the
    component sits); tied parameters follow their reference.
    """
    spec = copy.deepcopy(spec)
    comp = next((c for c in spec["components"] if c["name"] == name), None)
    if comp is None:
        raise KeyError(f"no component {name!r}")
    p = comp["params"]
    new = {"x0": x0, "y0": y0, "R_e": R_e}
    if q is not None or pa_deg is not None:
        if "e1" in p:
            q0, pa0 = sersic_qpa({k: p[k]["value"] for k in ("e1", "e2")})
            q = float(q0) if q is None else q
            pa_deg = float(pa0) if pa_deg is None else pa_deg
            e1, e2 = e_from_qpa(min(max(q, Q_MIN), 1.0), pa_deg)
            new.update(e1=float(e1), e2=float(e2))
        else:
            if pa_deg is not None:
                lo, hi = p["pa_deg"].get("lo"), p["pa_deg"].get("hi")
                mid = 0.5 * (lo + hi) if lo is not None and hi is not None else p["pa_deg"]["value"]
                pa_deg = pa_deg + 180.0 * round((mid - pa_deg) / 180.0)
            new.update(q=None if q is None else min(max(q, Q_MIN), 1.0), pa_deg=pa_deg)
    for k, x in new.items():
        if x is not None:
            if k not in p:
                raise KeyError(f"{name}: no parameter {k!r}")
            p[k]["value"] = float(_into_bounds(float(x), p[k]))
    return spec
