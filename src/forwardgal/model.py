"""The forward model as one function of the free-parameter vector u.

    u --(ModelParams.unpack)--> parameters of every component
      --> one template per component (unit total flux; Sersic: rendered on the
          traced grid in its plane, then PSF-convolved; sky: 1; a point with a
          locked image position, e.g. the quasar: its PSF, made once by
          config.build and passed in ``point_templates``)
      + fixed templates (any other templates without parameters)
      --> amplitudes: a component with a flux parameter ('flux' or 'mag',
          F2, 2026-11-22) has it GIVEN by u; the others (old files) are
          solved by NNLS (fluxes >= 0, sky free) on the data minus the given
          components, with optional amplitude priors
      --> loss(u) = chi2 + amplitude-prior penalty - 2 log p(u)

``loss`` is minus twice the log-posterior (up to a constant) for Gaussian noise,
without the Jacobian term (that is only for sampling, M11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

# [M12e paused] from forwardgal.components import modes_cs, sersic_qpa, spiral_of
from forwardgal.components import bending_of, modes_cs, sersic_qpa  # [M12e stand-in]
from forwardgal.likelihood import design_matrix, profile_chi2
from forwardgal.profiles.sersic import unit_flux_ie
from forwardgal.render import RenderSettings, render_lensed_sersic


def component_template(c, v, grid, convolve, pixels, pixscale, settings):
    """Template of one component as the data see it: (ny, nx) on the render box.

    c : Component (kind "sersic" or "sky")
    v : dict of its parameter values (from ModelParams.unpack); may be traced
    grid, convolve, pixels, pixscale, settings : as in ForwardModel

    Sersic: unit total flux (rendered in its plane on the traced grid, zero
    outside ``pixels`` apart from exact pixels), then PSF-convolved. Sky: 1
    everywhere. Shared by ForwardModel (fitting) and forwardgal.live (GUI).
    """
    if c.kind == "sersic":
        q, pa = sersic_qpa(v)  # from (e1, e2) or (q, pa_deg)
        modes = modes_cs(v, c.modes)  # None without Fourier modes
        ie = unit_flux_ie(v["R_e"], v["n"], q, pixscale, modes=modes)
        img = render_lensed_sersic(grid, v["x0"], v["y0"], ie, v["R_e"], v["n"], q,
                                   pa, settings=settings, pixels=pixels,
                                   # [M12e paused] plane=c.plane, modes=modes, spiral=spiral_of(v, c.spiral_kind))
                                   plane=c.plane, modes=modes, bending=bending_of(v, c))  # [M12e stand-in]
        return convolve(img)
    if c.kind == "sky":
        return jnp.ones(grid.shape)
    raise NotImplementedError(f"{c.name}: no template for kind {c.kind!r}")


@dataclass
class ForwardModel:
    """Everything needed to evaluate the model on one render box.

    Attributes
    ----------
    params : ModelParams
        Components (Sersic in either plane, sky, point) and their free
        parameters. A point needs its template in ``point_templates`` (its
        position is locked, so the template does not depend on u).
    grid : TracedGrid
        The render box, traced once (its subsample must match ``settings``).
    convolve : callable
        PSF convolution on the box (psf.make_convolver).
    data, sigma : (ny, nx) arrays
        Data and 1-sigma noise on the box.
    mask : (ny, nx) bool
        Fit pixels (only these enter chi^2).
    pixels : (rows, cols)
        Render pixels within the box (mask + margin): where Sersic templates
        are computed (zero elsewhere, apart from the exact pixels).
    pixscale : float
        arcsec per pixel (for the unit-flux normalization).
    point_templates : dict point-component name -> (ny, nx) array
        Unit-flux images of the point components (quasar.place_point), used
        in the components' order.
    fixed_templates : dict name -> (ny, nx) array
        Other templates without free parameters, already as the data see
        them (after the components).
    nonneg : dict name -> bool
        Which amplitudes must be >= 0 (default: all except the sky).
    amp_priors : dict name -> (mu, s)
        Gaussian amplitude priors (e.g. likelihood.flux_prior).
    method : str
        Amplitude solver ("nnls", "lstsq", "ridge") for components without
        a flux parameter.
    settings : RenderSettings
    zeropoint : float or None
        AB zeropoint, needed by 'mag' flux parameters (photometry.py).
    """

    params: object
    grid: object
    convolve: object
    data: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    pixels: tuple
    pixscale: float
    point_templates: dict = field(default_factory=dict)
    fixed_templates: dict = field(default_factory=dict)
    nonneg: dict = field(default_factory=dict)
    amp_priors: dict = field(default_factory=dict)
    method: str = "nnls"
    settings: RenderSettings = field(default_factory=RenderSettings)
    zeropoint: float | None = None

    def __post_init__(self):
        for c in self.params.components:
            if c.flux_key == "mag" and self.zeropoint is None:
                raise ValueError(f"{c.name}: a 'mag' flux parameter needs the image's zeropoint")
            if c.kind == "point" and c.name not in self.point_templates:
                raise NotImplementedError(f"{c.name}: a point needs its template in point_templates "
                                          "(free point positions are not supported yet)")
        self.names = [c.name for c in self.params.components] + list(self.fixed_templates)
        # Amplitudes GIVEN by a flux parameter, and the ones SOLVED (NNLS): indices into names.
        self._given = [i for i, c in enumerate(self.params.components) if c.flux_key is not None]
        self._solved = [i for i in range(len(self.names)) if i not in self._given]
        solved = [self.names[i] for i in self._solved]
        self.solved_names = solved  # amplitudes solved by NNLS (components without a flux parameter)
        self.given_names = [self.names[i] for i in self._given]  # amplitudes given by a flux parameter
        self._nonneg = [self.nonneg.get(n, self._kind(n) != "sky") for n in solved]
        mu = [self.amp_priors.get(n, (np.nan, np.nan))[0] for n in solved]
        sd = [self.amp_priors.get(n, (np.nan, np.nan))[1] for n in solved]
        self._prior_mu, self._prior_sd = jnp.array(mu), jnp.array(sd)
        self._d = jnp.asarray(np.asarray(self.data)[self.mask])
        self._s = jnp.asarray(np.asarray(self.sigma)[self.mask])
        self._has_priors = bool(np.isfinite(mu).any())
        self.loss = jax.jit(self._loss)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._loss))
        self.solve_jit = jax.jit(self.solve)

    def _kind(self, name):
        c = self.params.by_name.get(name)
        return c.kind if c is not None else "fixed"

    def templates(self, u):
        """(K, ny, nx) templates in ``self.names`` order (unit total flux each)."""
        values = self.params.unpack(u)
        out = [jnp.asarray(self.point_templates[c.name]) if c.kind == "point" else
               component_template(c, values[c.name], self.grid, self.convolve, self.pixels, self.pixscale,
                                  self.settings) for c in self.params.components]
        out += [jnp.asarray(t) for t in self.fixed_templates.values()]
        return jnp.stack(out)

    def given_amplitudes(self, values):
        """Amplitudes of the components with a flux parameter, in ``_given`` order (JAX).

        'flux': the value itself (sky: per pixel). 'mag': F = 10^(-0.4 (m - ZP));
        a sky in 'mag' is a surface brightness, s = pixscale^2 10^(-0.4 (mu - ZP)).
        See photometry.py.
        """
        out = []
        for i in self._given:
            c = self.params.components[i]
            x = values[c.name][c.flux_key]
            if c.flux_key == "mag":
                x = 10.0 ** (-0.4 * (x - self.zeropoint)) * (self.pixscale**2 if c.kind == "sky" else 1.0)
            out.append(x)
        return jnp.stack(out) if out else jnp.zeros(0)

    def _all_amplitudes(self, u, a_solved):
        """Amplitudes in ``names`` order: the given ones from u, the solved ones from a_solved."""
        a = jnp.zeros(len(self.names))
        if self._given:
            a = a.at[jnp.array(self._given)].set(self.given_amplitudes(self.params.unpack(u)))
        if self._solved:
            a = a.at[jnp.array(self._solved)].set(a_solved)
        return a

    def solve(self, u):
        """(loss terms, amplitudes): chi2 + penalty, and the amplitudes (in ``names`` order).

        The given components are subtracted from the data; the rest are
        solved (NNLS by default) on what remains. With no amplitude left to
        solve this is just chi^2 of the model with the given fluxes.
        """
        T_all = self.templates(u)
        T = design_matrix(T_all, self.mask)
        d = self._d
        if self._given:
            g = jnp.array(self._given)
            d = d - T[:, g] @ self.given_amplitudes(self.params.unpack(u))
        if not self._solved:
            return jnp.sum((d / self._s) ** 2), self._all_amplitudes(u, None)
        pm = self._prior_mu if self._has_priors else None
        ps = self._prior_sd if self._has_priors else None
        c2, a = profile_chi2(T[:, jnp.array(self._solved)], d, self._s, method=self.method, nonneg=self._nonneg,
                             prior_mu=pm, prior_sigma=ps)
        return c2, self._all_amplitudes(u, a)

    def _loss(self, u):
        c2, _ = self.solve(u)
        return c2 - 2.0 * self.params.log_prior(u, jacobian=False)

    def model_image(self, u):
        """Best-fit model image on the box (amplitudes solved at u)."""
        _, a = self.solve(u)
        return jnp.tensordot(a, self.templates(u), axes=1)

    def chi2_dof(self, u):
        """chi^2 in the mask and nu = N_mask - solved amplitudes - free parameters (fluxes included)."""
        T = design_matrix(self.templates(u), self.mask)
        _, a = self.solve(u)
        c2 = float(jnp.sum(((self._d - T @ a) / self._s) ** 2))
        dof = int(self.mask.sum()) - len(self._solved) - self.params.size
        return c2, dof

    def joint_loss(self, u, a):
        """Loss with the amplitudes as explicit parameters (not solved):

            chi^2(u, a) + amplitude-prior penalty(a) - 2 log p(u).

        At a = a*(u) it equals ``loss(u)``. Used for the Laplace approximation,
        where shapes and amplitudes are treated together. ``a`` has all
        amplitudes (``names`` order); the given ones are taken from u instead.
        """
        from forwardgal.likelihood import amplitude_penalty

        a = self._all_amplitudes(u, a[jnp.array(self._solved)] if self._solved else None)
        T = design_matrix(self.templates(u), self.mask)
        c2 = jnp.sum(((self._d - T @ a) / self._s) ** 2)
        a_s = a[jnp.array(self._solved)] if self._solved else a[:0]
        pen = amplitude_penalty(a_s, self._prior_mu, self._prior_sd) if self._has_priors else 0.0
        return c2 + pen - 2.0 * self.params.log_prior(u, jacobian=False)

    def joint_loss_solved(self, u, a_solved):
        """``joint_loss`` with only the SOLVED amplitudes explicit (``solved_names`` order).

        The given ones come from u. Laplace errors and NUTS use this, so a
        flux that is a parameter is not counted twice.
        """
        a = self._all_amplitudes(u, a_solved if self._solved else None)
        return self.joint_loss(u, a)

    def residuals(self, u, a):
        """All weighted residuals at shape parameters u and FIXED amplitudes a.

        [(d - T a) / sigma in the mask, shape-prior residuals, amplitude-prior
        residuals]; their sum of squares equals ``joint_loss(u, a)``. Used by
        Levenberg-Marquardt (Jacobian w.r.t. u with a held fixed). The given
        amplitudes are taken from u (their derivatives are part of the Jacobian).
        """
        a = self._all_amplitudes(u, a[jnp.array(self._solved)] if self._solved else None)
        T = design_matrix(self.templates(u), self.mask)
        parts = [(self._d - T @ a) / self._s, self.params.prior_residuals(u)]
        if self._has_priors:
            a = a[jnp.array(self._solved)]
            has = jnp.isfinite(self._prior_mu) & jnp.isfinite(self._prior_sd)
            mu = jnp.where(has, self._prior_mu, 0.0)
            sd = jnp.where(has, self._prior_sd, 1.0)
            parts.append(jnp.where(has, (a - mu) / sd, 0.0))
        return jnp.concatenate(parts)
