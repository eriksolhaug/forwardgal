"""NUTS sampling of a ForwardModel's posterior (numpyro).

Sampled vector z = (u, v):
  u : shape parameters, unconstrained (forwardgal.params); the Jacobian
      log|dx/du| is included, so bounds and transforms are handled correctly;
  v : amplitudes (SAMPLED, not solved, so flux uncertainties include all
      correlations): a_k = s_k softplus(v_k) for fluxes (>= 0), a_k = s_k v_k
      for the sky (free sign); s_k = scale from the best fit, so all numbers
      are of order 1. Flat priors on a_k unless amplitude priors are set.

Potential (minus log posterior, up to a constant):

    U(z) = 1/2 joint_loss(u, a) - log|dx/du| - sum_k log|da_k / dv_k|,

joint_loss = chi^2 + amplitude-prior penalty - 2 log p(u). NUTS (No-U-Turn
Sampler, Hamiltonian Monte Carlo) uses the exact gradient of U; with
dense_mass=True it learns the full parameter covariance during warm-up.

Diagnostics: R-hat (split Gelman-Rubin; ~1.00 = chains agree), effective
sample size, number of divergent transitions (should be 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import softplus

from forwardgal.components import aphi_from_cs, galfit_bend_sign, qpa_from_e, sersic_qpa


@dataclass
class Posterior:
    """Posterior samples in physical units.

    Attributes
    ----------
    names : list of str
        "Comp.param" (free shape parameters), "Comp.q" / "Comp.pa_deg" (derived,
        for (e1, e2) Sersics), "amp:Comp" (amplitudes / fluxes).
    chains : (n_chains, n_samples, P) array
    r_hat, n_eff : (P,) arrays
    divergences : int
    seconds : float
    """

    names: list
    chains: np.ndarray
    r_hat: np.ndarray
    n_eff: np.ndarray
    divergences: int
    seconds: float

    @property
    def samples(self):
        return self.chains.reshape(-1, self.chains.shape[-1])

    def column(self, name):
        return self.samples[:, self.names.index(name)]

    def describe(self, zeropoint=None):
        n_chains, n_samples, _ = self.chains.shape
        lines = [f"NUTS: {n_chains} chains x {n_samples} samples in {self.seconds:.0f} s; "
                 f"divergences: {self.divergences}",
                 f"{'parameter':28s} {'median':>11s} {'-1sigma':>9s} {'+1sigma':>9s} {'R-hat':>7s} {'n_eff':>7s}"]
        for k, n in enumerate(self.names):
            x = self.samples[:, k]
            lo, med, hi = np.percentile(x, [15.865, 50, 84.135])
            line = f"{n:28s} {med:11.5g} {med - lo:9.3g} {hi - med:9.3g} {self.r_hat[k]:7.3f} {self.n_eff[k]:7.0f}"
            if n.startswith("amp:") and zeropoint is not None and not n.endswith("sky") and np.all(x > 0):
                m = -2.5 * np.log10(x) + zeropoint
                ml, mm, mh = np.percentile(m, [15.865, 50, 84.135])
                line += f"   m = {mm:.3f} -{mm - ml:.3f} +{mh - mm:.3f}"
            lines.append(line)
        bad = [n for n, r in zip(self.names, self.r_hat) if r > 1.05]
        if bad:
            lines.append(f"WARNING: R-hat > 1.05 (chains disagree) for {bad}")
        if self.divergences:
            lines.append(f"WARNING: {self.divergences} divergent transitions: posterior geometry is difficult")
        return "\n".join(lines)

    def corner(self, keys=None, truths=None, laplace=None, path=None):
        """Corner plot of the chosen parameters (default all), optionally with the
        true values (dict name -> value) and the Laplace 1-sigma ellipses."""
        import corner as corner_lib
        import matplotlib.pyplot as plt

        keys = keys or self.names
        idx = [self.names.index(k) for k in keys]
        data = self.samples[:, idx]
        fig = corner_lib.corner(data, labels=keys, quantiles=[0.15865, 0.5, 0.84135], show_titles=True,
                                title_fmt=".4g", truths=[truths.get(k) if truths else None for k in keys],
                                truth_color="C3", label_kwargs=dict(fontsize=8), title_kwargs=dict(fontsize=7))
        if laplace is not None:
            axes = np.array(fig.axes).reshape(len(keys), len(keys))
            t = np.linspace(0, 2 * np.pi, 100)
            for r, kr in enumerate(keys):
                for c, kc in enumerate(keys[:r]):
                    if kr in laplace.names and kc in laplace.names:
                        i, j = laplace.names.index(kc), laplace.names.index(kr)
                        C = laplace.covariance[np.ix_([i, j], [i, j])]
                        vals, vecs = np.linalg.eigh(C)
                        ell = vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
                        axes[r, c].plot(laplace.best[i] + ell[0], laplace.best[j] + ell[1], "C2-", lw=1.2)
            fig.axes[0].plot([], [], "C2-", label="Laplace 1 sigma")
            fig.axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8)
        if path is not None:
            fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return fig


def nuts(model, result, num_warmup=500, num_samples=1000, num_chains=2, seed=0, dense_mass=True,
         target_accept=0.8, progress=True, max_tree_depth=10, adapt_mass=False):
    """Sample the posterior of ``model`` with NUTS, starting at the fit ``result``.

    The initial (inverse) mass matrix is the inverse curvature of the potential
    at the best fit (the Laplace covariance in the sampled variables), so NUTS
    starts with the right scale and correlations for every parameter (they
    span ~1e-4 to ~1 here; with an identity mass matrix NUTS needed ~1000
    gradients per step). Warm-up then refines it.

    progress : show numpyro's progress bar (recommended for long runs).
    adapt_mass : keep the Laplace mass matrix fixed (False, default) or let
        numpyro re-estimate it during warm-up (True). Measured on the
        synthetic setup (2026-11-16): fixed -> 7 gradients per step, 0
        divergences; re-estimated -> ~700 gradients per step. Any fixed mass
        matrix gives correct samples; a good one makes sampling efficient.
    Returns a Posterior. Chains run one after another (memory-friendly).
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS

    params = model.params
    nu = params.size
    a0 = np.array([result.amplitudes[n] for n in model.solved_names])  # given fluxes are in u (F3)
    nonneg = np.array(model._nonneg)
    scale = np.maximum(np.abs(a0), 1e-3)
    s = jnp.asarray(scale)
    nn = jnp.asarray(nonneg)

    def amplitudes(v):
        return jnp.where(nn, s * softplus(v), s * v)

    def log_jac_amp(v):
        # d a / d v = s sigmoid(v) for fluxes, s for the sky; log sigmoid(v) = -softplus(-v)
        return jnp.sum(jnp.where(nn, jnp.log(s) - softplus(-v), jnp.log(s)))

    def potential(z):
        u, v = z[:nu], z[nu:]
        a = amplitudes(v)
        log_jac_u = params.log_prior(u, jacobian=True) - params.log_prior(u, jacobian=False)
        return 0.5 * model.joint_loss_solved(u, a) - log_jac_u - log_jac_amp(v)

    # Starting point: the best fit (amplitudes inverted through the transforms).
    a_start = np.where(nonneg, np.maximum(a0, 1e-3 * scale), a0) / scale
    v0 = np.where(nonneg, a_start + np.log(-np.expm1(-a_start)), a_start)  # inverse softplus
    z0 = np.concatenate([np.asarray(result.u), v0])
    rng = np.random.default_rng(seed)
    z0_chains = jnp.asarray(z0[None, :] + 1e-3 * rng.normal(size=(num_chains, z0.size)))
    if num_chains == 1:  # numpyro wants a plain vector for a single chain
        z0_chains = z0_chains[0]

    # Initial inverse mass matrix: inverse Hessian of the potential at the best
    # fit (central differences of the exact gradient, symmetrized; negative or
    # tiny eigenvalues clipped so it is positive definite).
    grad = jax.jit(jax.grad(potential))
    zc = jnp.asarray(z0)
    P = z0.size
    H = np.zeros((P, P))
    for j in range(P):
        h = 1e-5 * max(1.0, abs(z0[j]))
        e = jnp.zeros(P).at[j].set(h)
        H[:, j] = (np.asarray(grad(zc + e)) - np.asarray(grad(zc - e))) / (2 * h)
    H = 0.5 * (H + H.T)
    vals, vecs = np.linalg.eigh(H)
    vals = np.maximum(vals, 1e-6 * vals.max())
    inv_mass = (vecs / vals) @ vecs.T
    if not dense_mass:
        inv_mass = np.diag(inv_mass)

    kernel = NUTS(potential_fn=potential, dense_mass=dense_mass, target_accept_prob=target_accept,
                  inverse_mass_matrix=jnp.asarray(inv_mass), max_tree_depth=max_tree_depth,
                  adapt_mass_matrix=adapt_mass)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=progress)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), init_params=z0_chains, extra_fields=("diverging",))
    z = np.asarray(mcmc.get_samples(group_by_chain=True))  # (chains, samples, P)
    seconds = time.perf_counter() - t0
    divergences = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())

    # Physical values for every sample.
    e_comps = [c.name for c in params.components if c.kind == "sersic"
               and f"{c.name}.e1" in params.free_keys and f"{c.name}.e2" in params.free_keys]
    mode_comps = [(c.name, m) for c in params.components for m in c.modes
                  if f"{c.name}.c{m}" in params.free_keys and f"{c.name}.s{m}" in params.free_keys]
    bend_comps = [(c.name, m, c.bend_pa_ref) for c in params.components for m in c.bends
                  if f"{c.name}.b{m}" in params.free_keys]
    # [M12e paused] spiral_comps = [c.name for c in params.components if c.spiral_kind is not None]

    def physical(zz):
        u, v = zz[:nu], zz[nu:]
        vals = params.unpack(u)
        shape = [vals[k.split(".")[0]][k.split(".")[1]] for k in params.free_keys]
        derived = []
        for name in e_comps:
            q, pa = qpa_from_e(vals[name]["e1"], vals[name]["e2"])
            derived += [q, jnp.mod(pa, 180.0)]
        for name, m in mode_comps:
            derived += list(aphi_from_cs(vals[name][f"c{m}"], vals[name][f"s{m}"], m))
        for name, m, pa_ref in bend_comps:
            derived.append(vals[name][f"b{m}"] * galfit_bend_sign(sersic_qpa(vals[name])[1], pa_ref, m))
        # [M12e paused] for name in spiral_comps:
            # [M12e paused] derived.append(vals[name]["sp_r_in"] + vals[name]["sp_dr"])
        return jnp.concatenate([jnp.stack(shape) if shape else jnp.zeros(0),
                                jnp.stack(derived) if derived else jnp.zeros(0), amplitudes(v),
                                model.given_amplitudes(vals)])

    phys = np.asarray(jax.vmap(jax.vmap(physical))(jnp.asarray(z)))
    names = (list(params.free_keys) + [f"{n}.{p}" for n in e_comps for p in ("q", "pa_deg")]
             + [f"{n}.{p}{m}" for n, m in mode_comps for p in ("a", "phi")]
             + [f"{n}.b{m}_galfit" for n, m, _ in bend_comps]
             # [M12e paused] + [f"{n}.sp_r_out" for n in spiral_comps]
             + [f"amp:{n}" for n in model.solved_names] + [f"amp:{n}" for n in model.given_names])
    r_hat = np.asarray(split_gelman_rubin(phys)) if num_chains > 1 else np.full(len(names), np.nan)
    n_eff = np.asarray(effective_sample_size(phys))
    return Posterior(names, phys, r_hat, n_eff, divergences, seconds)
