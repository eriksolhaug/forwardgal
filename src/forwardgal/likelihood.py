"""chi^2 and linear amplitudes.

The model is m = sum_k a_k T_k: each component k has a TEMPLATE T_k (its model
image at unit brightness, as the data see it: Sersic templates are
PSF-convolved, quasar templates are unit-flux PSFs, the sky template is 1)
and a linear amplitude a_k. Within the mask, with data d and noise sigma,

    chi^2(a) = sum_i ((d_i - sum_k T_ik a_k) / sigma_i)^2,

which is minimized over a directly (weighted linear least squares):

- "lstsq"  : unconstrained; covariance (T^T W T)^-1 with W = diag(1/sigma^2).
- "nnls"   : amplitudes flagged non-negative are constrained >= 0 (fluxes);
             others (e.g. the sky) are free in sign. scipy.optimize.nnls.
- "ridge"  : unconstrained + lambda sum_k (a_k / s_k)^2 with s_k = the
             column scale, to stabilize nearly degenerate components.

Amplitude priors: a Gaussian prior a_k ~ N(mu_k, s_k) adds ((a_k - mu_k)/s_k)^2
to chi^2. That is one extra least-squares row (e_k / s_k with value mu_k / s_k),
so amplitudes are still solved directly, with any method. Covariance with
priors: (T^T W T + diag(1/s_k^2))^-1. Use NaN in prior_mu / prior_sigma for
"no prior" on an amplitude. Helpers: ``flux_prior``, ``magnitude_prior``.

Gradients (envelope theorem): with chi2_min(p) = min_a chi^2(p, a),
d chi2_min / dp = partial chi^2 / partial p at a = a*(p), because
partial chi^2 / partial a = 0 at the optimum (also for NNLS: amplitudes
pinned at 0 stay there for small changes of p). ``profile_chi2`` therefore
treats a* as a constant (jax.lax.stop_gradient), so no solver needs to be
differentiated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls


def design_matrix(templates, mask):
    """(K, ny, nx) templates -> (N_mask, K) matrix of their values in the mask."""
    return jnp.stack([t[mask] for t in templates], axis=1)


def _prior_rows(K, prior_mu, prior_sigma):
    """Extra least-squares rows (P, K) and values (P,) for the amplitude priors."""
    if prior_mu is None:
        return np.zeros((0, K)), np.zeros(0)
    mu, s = np.asarray(prior_mu, float), np.asarray(prior_sigma, float)
    has = np.isfinite(mu) & np.isfinite(s)
    rows = np.zeros((has.sum(), K))
    rows[np.arange(has.sum()), np.nonzero(has)[0]] = 1.0 / s[has]
    return rows, mu[has] / s[has]


def _solve_numpy(T, d, sigma, method, nonneg, ridge, prior_mu=None, prior_sigma=None):
    Tw = np.asarray(T) / np.asarray(sigma)[:, None]
    dw = np.asarray(d) / np.asarray(sigma)
    K = Tw.shape[1]
    rows, vals = _prior_rows(K, prior_mu, prior_sigma)
    Tw, dw = np.vstack([Tw, rows]), np.concatenate([dw, vals])
    if not (np.all(np.isfinite(Tw)) and np.all(np.isfinite(dw))):
        # e.g. an optimizer trial step to extreme parameters: return NaN so the
        # loss is NaN and the step is rejected (instead of the solver raising).
        return np.full(K, np.nan)
    if method == "lstsq":
        return np.linalg.lstsq(Tw, dw, rcond=None)[0]
    if method == "ridge":
        scale = np.linalg.norm(Tw, axis=0)
        scale[scale == 0] = 1.0
        reg = np.sqrt(ridge) * np.diag(scale)  # penalty lambda * sum (a_k * |T_k|)^2
        return np.linalg.lstsq(np.vstack([Tw, reg]), np.concatenate([dw, np.zeros(K)]), rcond=None)[0]
    if method == "nnls":
        # Free-sign amplitudes are split into a positive and a negative part.
        nonneg = np.ones(K, bool) if nonneg is None else np.asarray(nonneg, bool)
        cols, back = [], []
        for k in range(K):
            cols.append(Tw[:, k])
            back.append((k, 1.0))
            if not nonneg[k]:
                cols.append(-Tw[:, k])
                back.append((k, -1.0))
        x, _ = nnls(np.column_stack(cols), dw, maxiter=50 * len(cols))
        a = np.zeros(K)
        for xi, (k, sgn) in zip(x, back):
            a[k] += sgn * xi
        return a
    raise ValueError(f"unknown method {method!r}")


def solve_amplitudes(T, d, sigma, method="lstsq", nonneg=None, ridge=1e-6, prior_mu=None, prior_sigma=None):
    """Best amplitudes a* (K,) for design matrix T (N, K), data d (N,), noise sigma (N,).

    Works on numpy or JAX arrays; inside jax.jit it runs via jax.pure_callback
    (it is not differentiated: see ``profile_chi2``).

    nonneg : (K,) bool, for method "nnls": which amplitudes must be >= 0
             (default: all).
    ridge  : lambda for method "ridge" (dimensionless, relative to each
             column's weighted norm).
    prior_mu, prior_sigma : (K,) arrays or None
        Gaussian amplitude priors; NaN = no prior for that amplitude.
    """
    K = T.shape[1]
    out = jax.ShapeDtypeStruct((K,), jnp.float64)
    nan = jnp.full(K, jnp.nan)
    mu = nan if prior_mu is None else jnp.asarray(prior_mu, dtype=float)
    sd = nan if prior_sigma is None else jnp.asarray(prior_sigma, dtype=float)
    return jax.pure_callback(
        lambda T_, d_, s_, m_, p_: _solve_numpy(T_, d_, s_, method, nonneg, ridge, m_, p_).astype(np.float64),
        out, T, d, sigma, mu, sd,
    )


def chi2(T, a, d, sigma):
    """chi^2 = sum ((d - T a) / sigma)^2."""
    return jnp.sum(((d - T @ a) / sigma) ** 2)


def amplitude_penalty(a, prior_mu, prior_sigma):
    """sum_k ((a_k - mu_k) / s_k)^2 over the amplitudes that have a prior."""
    if prior_mu is None:
        return 0.0
    mu, s = jnp.asarray(prior_mu, dtype=float), jnp.asarray(prior_sigma, dtype=float)
    has = jnp.isfinite(mu) & jnp.isfinite(s)
    z = jnp.where(has, (a - jnp.where(has, mu, 0.0)) / jnp.where(has, s, 1.0), 0.0)
    return jnp.sum(z**2)


def profile_chi2(T, d, sigma, method="lstsq", nonneg=None, ridge=1e-6, prior_mu=None, prior_sigma=None):
    """chi^2 (+ amplitude-prior penalty) minimized over the amplitudes.

    Returns (chi2_min + penalty, a*). Without amplitude priors this is just
    chi2_min. a* is held constant for differentiation (envelope theorem; see
    module docstring), so gradients w.r.t. T (and prior_mu, prior_sigma) are
    correct.
    """
    # Solve on gradient-stopped inputs: JAX never differentiates the solver
    # (which runs outside JAX for NNLS), and a* enters chi^2 as a constant.
    sg = jax.lax.stop_gradient
    mu = None if prior_mu is None else sg(jnp.asarray(prior_mu, dtype=float))
    sd = None if prior_sigma is None else sg(jnp.asarray(prior_sigma, dtype=float))
    a = solve_amplitudes(sg(T), sg(d), sg(sigma), method, nonneg, ridge, mu, sd)
    return chi2(T, a, d, sigma) + amplitude_penalty(a, prior_mu, prior_sigma), a


def amplitude_covariance(T, sigma, prior_sigma=None):
    """Covariance of unconstrained amplitudes: (T^T W T + diag(1/s^2))^-1.

    W = diag(1/sigma^2); prior_sigma (K,) adds the Gaussian amplitude priors
    (NaN = none).
    """
    Tw = np.asarray(T) / np.asarray(sigma)[:, None]
    F = Tw.T @ Tw
    if prior_sigma is not None:
        s = np.asarray(prior_sigma, float)
        F = F + np.diag(np.where(np.isfinite(s), 1.0 / np.where(np.isfinite(s), s, 1.0) ** 2, 0.0))
    return np.linalg.inv(F)


def flux_prior(ref_flux, frac):
    """(mu, s) for "total flux within frac of a reference", e.g. +-20% of F110W."""
    return float(ref_flux), float(frac) * abs(float(ref_flux))


def magnitude_prior(m_ref, sigma_mag, zeropoint):
    """(mu, s) in flux for a Gaussian prior m ~ N(m_ref, sigma_mag), to FIRST ORDER.

    Flux F = 10^(-0.4 (m - ZP)) (pixel-sum units); dF / F = 0.4 ln(10) dm, so
    s = 0.4 ln(10) sigma_mag F_ref. Accurate to a few per cent for
    sigma_mag <~ 0.2 mag. An exact magnitude prior requires sampling the
    amplitudes (M11, NUTS).
    """
    F = 10.0 ** (-0.4 * (m_ref - zeropoint))
    return F, 0.4 * np.log(10.0) * sigma_mag * F
