"""Optimization of a ForwardModel: Levenberg-Marquardt (default) or L-BFGS.

- "lm": Levenberg-Marquardt, as in GALFIT, for our sum-of-squares loss
  L(u) = sum_i r_i(u)^2. Each iteration solves
      (J^T J + lambda diag(J^T J)) delta = -J^T r,   J = dr/du (exact, JAX),
  with the amplitudes held at their solved values while J is computed
  (variable projection); lambda shrinks x10 after a successful step and grows
  x10 after a rejected one. Converges in tens of iterations.
- "lbfgs": L-BFGS (optax), a general quasi-Newton method with a line search;
  kept for comparison.

The fit runs from the starting values and from ``n_starts - 1`` randomly
perturbed copies (perturbations in PHYSICAL units, per parameter type; see
JITTER), and keeps the lowest loss. Every start is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from forwardgal.components import aphi_from_cs, galfit_bend_sign, qpa_from_e, sersic_qpa

# Default random perturbation of starting values, by parameter name:
# ("add", sigma) adds N(0, sigma); ("mul", f) multiplies by (1 + N(0, f)).
JITTER = {
    "x0": ("add", 0.03), "y0": ("add", 0.03),  # arcsec
    "R_e": ("mul", 0.3), "n": ("add", 0.5), "q": ("add", 0.1), "pa_deg": ("add", 20.0),
    "e1": ("add", 0.1), "e2": ("add", 0.1),
    "b2": ("add", 0.01), "b3": ("add", 0.01), "b4": ("add", 0.01), "b5": ("add", 0.01),  # bending (arcsec)
    "mag": ("add", 0.2), "flux": ("mul", 0.2),  # flux parameters (F3): +-0.2 mag, +-20 %
    # [M12e paused] "sp_r_in": ("mul", 0.2), "sp_dr": ("mul", 0.2), "sp_theta": ("add", 20.0),  # spiral (deg for theta)
    # [M12e paused] "sp_alpha": ("add", 0.2), "sp_r_ws": ("mul", 0.3),
}
W_JITTER = 0.2  # jitter of jointly bounded pairs, in the dimensionless w (|w| < 1)
MODE_JITTER = 0.03  # jitter of Fourier (c_m, s_m), added in physical units


@dataclass
class FitResult:
    u: np.ndarray
    loss: float
    values: dict
    amplitudes: dict
    starts: list  # per start: (loss, n_iterations, converged)
    seconds: float

    def describe(self, model):
        c2, dof = model.chi2_dof(jnp.asarray(self.u))
        lines = [f"best loss {self.loss:.2f}; chi2/dof = {c2:.1f}/{dof} = {c2 / dof:.3f}; "
                 f"{len(self.starts)} start(s) in {self.seconds:.0f} s"]
        for k, (loss, it, conv) in enumerate(self.starts):
            lines.append(f"  start {k}: loss {loss:12.2f} after {it:4d} iterations{'' if conv else ' (NOT converged)'}")
        for c in model.params.components:
            v = self.values[c.name]
            if v:
                lines.append(f"{c.name} ({c.plane} plane): " + ", ".join(f"{k}={float(x):.4g}" for k, x in v.items()))
        for n, a in self.amplitudes.items():
            lines.append(f"amplitude {n}: {a:.5g}")
        return "\n".join(lines)


def _with_derived(params, values):
    """Add q and pa_deg (in [0, 180)) for Sersics parametrized by (e1, e2), and
    GALFIT's a_m, phi_m (phi in [0, 360/m)) for Cartesian Fourier modes."""
    out = {k: dict(v) for k, v in values.items()}
    for c in params.components:
        if c.kind == "sersic" and "e1" in c.params:
            q, pa = qpa_from_e(out[c.name]["e1"], out[c.name]["e2"])
            out[c.name]["q"] = q
            out[c.name]["pa_deg"] = jnp.mod(pa, 180.0)
        for m in c.modes:
            if f"c{m}" in c.params:
                out[c.name][f"a{m}"], out[c.name][f"phi{m}"] = aphi_from_cs(out[c.name][f"c{m}"],
                                                                         out[c.name][f"s{m}"], m)
        for m in c.bends:  # GALFIT's sign convention (pa_ref = 0); needs the PA
            pa = out[c.name]["pa_deg"]
            out[c.name][f"b{m}_galfit"] = out[c.name][f"b{m}"] * galfit_bend_sign(pa, c.bend_pa_ref, m)
        # [M12e paused] if c.spiral_kind is not None:
            # [M12e paused] out[c.name]["sp_r_out"] = out[c.name]["sp_r_in"] + out[c.name]["sp_dr"]
    return out


def _jittered_start(params, rng):
    u = []
    for key in params.free_keys:
        p = params._lookup(key)
        how, size = JITTER.get(key.split(".")[1], ("add", 0.0))
        for _ in range(100):  # redraw until inside the bounds
            x = p.value + rng.normal(0, size) if how == "add" else p.value * (1 + rng.normal(0, size))
            if p.within_bounds(x):
                break
        else:
            x = p.value
        u.append(float(p.to_unconstrained(x)))
    u = jnp.array(u)
    # Jointly bounded pairs (|e| bound, "within" constraints): jitter inside the
    # allowed region, in w = u / sqrt(1 + |u|^2), starting from the valid start.
    u0 = params.start()
    for pc in params.constraints.values():
        i, j = params._pair_index(pc)
        if pc.mode != "bounds":
            u = u.at[i].set(u0[i]).at[j].set(u0[j])
            continue
        w = np.array([float(u0[i]), float(u0[j])]) / np.sqrt(1.0 + float(u0[i]) ** 2 + float(u0[j]) ** 2)
        w = w + rng.normal(0, W_JITTER, 2)
        r = np.hypot(*w)
        if r >= 0.95:
            w *= 0.95 / r
        un = w / np.sqrt(1.0 - (w**2).sum())
        u = u.at[i].set(un[0]).at[j].set(un[1])
    # Fourier modes under the joint bound: jitter (c_m, s_m) physically, keep
    # sum a_m <= 0.8 x limit, map back to the unconstrained w.
    for fb in params.fourier_bounds.values():
        idx = params._fourier_index(fb)
        y = np.asarray(fb.to_physical(u0[jnp.array(idx)])) + rng.normal(0, MODE_JITTER, len(idx))
        a_sum = np.hypot(y[0::2], y[1::2]).sum()
        if a_sum > 0.8 * fb.limit:
            y *= 0.8 * fb.limit / a_sum
        u = u.at[jnp.array(idx)].set(jnp.asarray(fb.to_unconstrained(y)))
    return u


_STEPS = {}  # one compiled L-BFGS step per model (id -> (optimizer, step))


def _lbfgs_step(model):
    """The optimizer and ONE compiled iteration (value, gradient, L-BFGS update).

    Compiled once per model and reused for all iterations and starts. (Calling
    optax's update uncompiled from a Python loop re-traces its line search,
    and recompiles the loss inside it, on every iteration: memory grows until
    it runs out, which is what crashed the first M10 test.)
    """
    key = id(model)
    if key not in _STEPS:
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(model._loss)

        @jax.jit
        def step(u, state):
            value, grad = vg(u, state=state)
            updates, state = opt.update(grad, state, u, value=value, grad=grad, value_fn=model._loss)
            return optax.apply_updates(u, updates), state, value, grad

        _STEPS[key] = (opt, step)
    return _STEPS[key]


def lbfgs(model, u0, max_iter=300, gtol=1e-4, ftol=1e-9, callback=None):
    """Minimize model.loss from u0. Returns (u, loss, iterations, converged).

    callback(iteration, loss), if given, is called after every iteration (e.g.
    for progress display or benchmarks).
    """
    opt, step = _lbfgs_step(model)
    state = opt.init(u0)
    u, prev = u0, np.inf
    for it in range(1, max_iter + 1):
        u, state, value, grad = step(u, state)
        value = float(value)
        if callback is not None:
            callback(it, value)
        if float(jnp.max(jnp.abs(grad))) < gtol or abs(prev - value) < ftol * max(1.0, abs(value)):
            return u, float(model.loss(u)), it, True
        prev = value
    return u, float(model.loss(u)), max_iter, False


_LM = {}  # compiled residuals and Jacobian per model (id -> (residuals, jacobian))


def levenberg_marquardt(model, u0, max_iter=100, ftol=1e-9, lam0=1e-3, callback=None, monitor=None):
    """Minimize model.loss from u0 with Levenberg-Marquardt.

    Returns (u, loss, iterations, converged). Converged = the relative loss
    decrease of an accepted step fell below ftol, or no step could reduce the
    loss (lambda grew past 1e10). NOT converged: max_iter reached, or the
    Jacobian is not finite (the model is undefined there; stops at once).
    callback(iteration, loss) is called after every iteration.
    monitor(iteration, loss, u, amplitudes, lam), if given, is called at the
    start of every iteration with the current state (progress printing).
    """
    key = id(model)
    if key not in _LM:
        _LM[key] = (jax.jit(model.residuals), jax.jit(jax.jacfwd(model.residuals, argnums=0)))
    res_fn, jac_fn = _LM[key]
    u = jnp.asarray(u0)
    loss = float(model.loss(u))
    lam = lam0
    for it in range(1, max_iter + 1):
        _, a = model.solve_jit(u)
        if monitor is not None:
            monitor(it, loss, u, a, lam)
        r = np.asarray(res_fn(u, a))
        J = np.asarray(jac_fn(u, a))
        if not (np.all(np.isfinite(J)) and np.all(np.isfinite(r))):
            return u, loss, it, False
        g, A = J.T @ r, J.T @ J
        D = np.diag(np.where(np.diag(A) > 0, np.diag(A), 1.0))
        while True:
            try:
                delta = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                delta = None
            new = None if delta is None else float(model.loss(u + delta))
            if new is not None and np.isfinite(new) and new < loss:
                u, decrease, loss = u + delta, loss - new, new
                lam = max(lam / 10.0, 1e-12)
                break
            lam *= 10.0
            if lam > 1e10:  # no direction reduces the loss: at a minimum
                if callback is not None:
                    callback(it, loss)
                return u, loss, it, True
        if callback is not None:
            callback(it, loss)
        if decrease < ftol * max(1.0, abs(loss)):
            return u, loss, it, True
    return u, loss, max_iter, False


def progress_printer(model, start, n_starts, every=1, file=None):
    """A ``monitor`` that prints the fit state GALFIT-style (see ``fit``)."""
    import sys

    out = file or sys.stdout
    params = model.params
    n_pix = int(np.asarray(model.mask).sum())
    dof = max(1, n_pix - len(model.names) - params.size)
    hidden = {"e1", "e2"} | {f"{p}{m}" for p in ("c", "s") for m in range(1, 10)}

    def fmt(c, key, x):
        prm = c.params.get(key)
        s = f"{x:.5g}"
        if prm is None:  # derived (q, pa_deg from e1, e2; a_m, phi_m from c_m, s_m)
            return s
        if not prm.free:
            return f"[{s}]"
        tol = 1e-6 * max(1.0, abs(x))
        if (prm.lo is not None and x - prm.lo < tol) or (prm.hi is not None and prm.hi - x < tol):
            return f"*{s}*"
        return s

    def show(it, loss, u, a, lam):
        if it != 1 and it % every:
            return
        vals = _with_derived(params, params.unpack(u))
        lam_s = "" if lam is None else f"   lambda = {lam:.0e}"
        lines = [f"start {start + 1}/{n_starts}  iteration {it:4d}   chi2/nu = {float(loss) / dof:.4f}{lam_s}"]
        for c in params.components:
            if c.kind == "sky":
                continue
            items = [(k, float(v)) for k, v in vals[c.name].items() if k not in hidden and not k.endswith("_galfit")]
            lines.append(f"  {c.name:8s} ({c.plane})  " + "  ".join(f"{k} {fmt(c, k, x)}" for k, x in items))
        if a is not None:
            lines.append("  fluxes  " + "  ".join(f"{n} {float(x):.5g}" for n, x in zip(model.names, np.asarray(a))))
        print("\n".join(lines), file=out, flush=True)

    return show


def fit(model, n_starts=1, seed=0, max_iter=None, method="lm", progress=False, every=1):
    """Multi-start fit. Returns the FitResult of the lowest loss.

    method : "lm" (Levenberg-Marquardt, default) or "lbfgs".
    max_iter : per start; default 100 for "lm", 300 for "lbfgs".
    progress : print the current parameters every ``every`` iterations,
        GALFIT-style (LM only): chi2/nu, lambda, every component's shape
        (q, PA and a_m, phi_m shown instead of e1, e2 and c_m, s_m), and the
        solved fluxes. *value* = a free parameter at its bound, [value] = fixed.
    """
    if method not in ("lm", "lbfgs"):
        raise ValueError(f"method must be 'lm' or 'lbfgs', got {method!r}")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    starts = [model.params.start()] + [_jittered_start(model.params, rng) for _ in range(n_starts - 1)]
    if method == "lm":
        results = [levenberg_marquardt(model, u, max_iter=max_iter or 100,
                                       monitor=progress_printer(model, k, n_starts, every) if progress else None)
                   for k, u in enumerate(starts)]
    else:
        results = [lbfgs(model, u, max_iter=max_iter or 300) for u in starts]
    best = min(results, key=lambda r: r[1])
    u = best[0]
    _, a = model.solve(u)
    return FitResult(
        u=np.asarray(u),
        loss=best[1],
        values=_with_derived(model.params, model.params.unpack(u)),
        amplitudes={n: float(x) for n, x in zip(model.names, np.asarray(a))},
        starts=[(r[1], r[2], r[3]) for r in results],
        seconds=time.perf_counter() - t0,
    )


@dataclass
class LaplaceErrors:
    """Approximate 1-sigma uncertainties from the curvature at the best fit.

    Attributes
    ----------
    names : list of str
        "Comp.param" for shape parameters (physical units), then
        "amp:Comp" for amplitudes.
    best : np.ndarray
        Best-fit values, same order.
    sigma : np.ndarray
        1-sigma errors (sqrt of the covariance diagonal).
    covariance, correlation : 2-D arrays
    scale : float
        sqrt(chi2/dof): multiply ``sigma`` by this for errors inflated by
        the model mismatch (reported separately).
    warnings : list of str
    """

    names: list
    best: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    scale: float
    warnings: list
    hessian: np.ndarray = None  # of joint_loss w.r.t. (u, amplitudes)
    derived: list = None  # (name, best, sigma) for derived quantities (q, PA from e1, e2)

    def describe(self, zeropoint=None):
        lines = [f"Laplace (curvature) errors; scaled column x sqrt(chi2/dof) = {self.scale:.3f}"]
        lines.append(f"{'parameter':28s} {'best':>12s} {'sigma':>10s} {'scaled':>10s}")
        for n, b, s in zip(self.names, self.best, self.sigma):
            line = f"{n:28s} {b:12.5g} {s:10.3g} {s * self.scale:10.3g}"
            if n.startswith("amp:") and zeropoint is not None and b > 0 and not n.endswith("sky"):
                line += f"   m = {-2.5 * np.log10(b) + zeropoint:.3f} +- {1.0857 * s / b:.3f}"
            lines.append(line)
        for n, b, sg in self.derived or []:
            lines.append(f"{n:28s} {b:12.5g} {sg:10.3g} {sg * self.scale:10.3g}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def laplace_errors(model, result):
    """Laplace approximation at the best fit: covariance = 2 H^-1 (loss = -2 log p).

    H is the Hessian of ``model.joint_loss_solved`` w.r.t. (u, solved
    amplitudes) together, so shape-amplitude correlations are included (a
    flux given by a parameter is part of u; its other unit, flux or mag, is
    added as a derived value). It is computed by central
    finite differences of the EXACT gradient (JAX cannot take a second
    derivative of the incomplete gamma function w.r.t. its shape argument, i.e.
    through the Sersic n), then symmetrized; relative accuracy ~1e-6. Shape errors are converted
    from u to physical units with the Jacobian dx/du (delta method).
    Approximate: assumes a near-Gaussian posterior. Unreliable for parameters
    near their bounds and for amplitudes pinned at 0 by NNLS (flagged).
    Errors are statistical, given the fixed lens model and PSF.
    """
    u = jnp.asarray(result.u)
    a = jnp.array([result.amplitudes[n] for n in model.solved_names])
    nu, na = len(u), len(a)
    z = jnp.concatenate([u, a])
    grad = jax.jit(jax.grad(lambda z: model.joint_loss_solved(z[:nu], z[nu:])))
    H = np.zeros((nu + na, nu + na))
    for j in range(nu + na):
        h = 1e-5 * max(1.0, abs(float(z[j])))
        e = jnp.zeros(nu + na).at[j].set(h)
        H[:, j] = (np.asarray(grad(z + e)) - np.asarray(grad(z - e))) / (2 * h)
    H = 0.5 * (H + H.T)
    warnings = []
    try:
        cov_z = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_z = 2.0 * np.linalg.pinv(H)
        warnings.append("Hessian is singular (a parameter is unconstrained); used a pseudo-inverse")
    if np.any(np.linalg.eigvalsh(0.5 * (H + H.T)) <= 0):
        warnings.append("Hessian is not positive definite: the fit may not be at a minimum")

    # Physical shape parameters: x = x(u) elementwise -> J diagonal (positions
    # under a joint "within" constraint are mapped by unpack; use autodiff).
    def physical(u_):
        v = model.params.unpack(u_)
        return jnp.array([v[k.split(".")[0]][k.split(".")[1]] for k in model.params.free_keys])

    J = np.asarray(jax.jacfwd(physical)(u))
    Jfull = np.zeros((nu + na, nu + na))
    Jfull[:nu, :nu] = J
    Jfull[nu:, nu:] = np.eye(na)
    cov = Jfull @ cov_z @ Jfull.T
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(np.where(sig > 0, sig, 1), np.where(sig > 0, sig, 1))

    names = list(model.params.free_keys) + [f"amp:{n}" for n in model.solved_names]
    best = np.concatenate([np.asarray(physical(u)), np.asarray(a)])
    # Derived q and PA for (e1, e2) Sersics, by error propagation (delta method).
    derived = []
    for c in model.params.components:
        k1, k2 = f"{c.name}.e1", f"{c.name}.e2"
        if c.kind == "sersic" and k1 in names and k2 in names:
            i, j = names.index(k1), names.index(k2)
            e = jnp.array([best[i], best[j]])
            conv = lambda e_: jnp.stack(qpa_from_e(e_[0], e_[1]))  # noqa: E731
            G = np.asarray(jax.jacfwd(conv)(e))
            C = cov[np.ix_([i, j], [i, j])]
            cq = G @ C @ G.T
            qv, pav = (float(x) for x in conv(e))
            derived += [(f"{c.name}.q (derived)", qv, float(np.sqrt(max(cq[0, 0], 0)))),
                        (f"{c.name}.pa_deg (derived)", pav % 180.0, float(np.sqrt(max(cq[1, 1], 0))))]
        # GALFIT a_m, phi_m from Cartesian Fourier modes (delta method; the
        # phase error is meaningless when a_m is within ~1 sigma of 0).
        for m in c.modes:
            k1, k2 = f"{c.name}.c{m}", f"{c.name}.s{m}"
            if k1 in names and k2 in names:
                i, j = names.index(k1), names.index(k2)
                conv = lambda y_, m=m: jnp.stack(aphi_from_cs(y_[0], y_[1], m))  # noqa: E731
                y = jnp.array([best[i], best[j]])
                G = np.asarray(jax.jacfwd(conv)(y))
                ca = G @ cov[np.ix_([i, j], [i, j])] @ G.T
                av, phv = (float(x) for x in conv(y))
                derived += [(f"{c.name}.a{m} (derived)", av, float(np.sqrt(max(ca[0, 0], 0)))),
                            (f"{c.name}.phi{m} (derived)", phv, float(np.sqrt(max(ca[1, 1], 0))))]
        # Bending amplitudes in GALFIT's sign convention (a sign flip: same sigma).
        vals_c = model.params.unpack(u)[c.name]
        pa_c = sersic_qpa(vals_c)[1] if c.kind == "sersic" else 0.0
        for m in c.bends:
            k = f"{c.name}.b{m}"
            if k in names:
                i = names.index(k)
                sgn = float(galfit_bend_sign(pa_c, c.bend_pa_ref, m))
                derived.append((f"{c.name}.b{m}_galfit (derived)", sgn * float(best[i]), float(sig[i])))
        # [M12e paused] # Spiral r_out = r_in + dr (both free: variance of the sum incl. covariance).
        # [M12e paused] k1, k2 = f"{c.name}.sp_r_in", f"{c.name}.sp_dr"
        # [M12e paused] if c.spiral_kind is not None and k1 in names and k2 in names:
            # [M12e paused] i, j = names.index(k1), names.index(k2)
            # [M12e paused] var = cov[i, i] + cov[j, j] + 2 * cov[i, j]
            # [M12e paused] derived.append((f"{c.name}.sp_r_out (derived)", float(best[i] + best[j]), float(np.sqrt(max(var, 0)))))
    # Flux parameters: the other unit as a derived value (delta method; m = -2.5 log10 F + ZP,
    # so sigma_m = 2.5 / ln 10 * sigma_F / F). A sky in mag is a surface brightness (photometry.py).
    k = 2.5 / np.log(10.0)
    for c in model.params.components:
        key = f"{c.name}.{c.flux_key}"
        if c.flux_key is None or key not in names or model.zeropoint is None:
            continue
        i = names.index(key)
        x, sx = float(best[i]), float(sig[i])
        scale = model.pixscale**2 if c.kind == "sky" else 1.0
        if c.flux_key == "mag":
            f = scale * 10.0 ** (-0.4 * (x - model.zeropoint))
            derived.append((f"{c.name}.flux (derived)", f, f * sx / k))
        elif x > 0:
            derived.append((f"{c.name}.mag (derived)", -2.5 * np.log10(x / scale) + model.zeropoint, k * sx / x))
    for n, nonneg in zip(model.solved_names, model._nonneg):
        if nonneg and result.amplitudes[n] <= 0:
            warnings.append(f"amplitude {n} is pinned at 0 by NNLS: its error is not meaningful")
    for k in model.params.free_keys:
        p = model.params._lookup(k)
        x = float(np.asarray(physical(u))[model.params.free_keys.index(k)])
        s = sig[model.params.free_keys.index(k)]
        tiny = 1e-6 * (abs(x) + 1.0)  # a parameter AT its bound has sigma ~ 0: flag it too
        if (p.lo is not None and x - p.lo < max(2 * s, tiny)) or (p.hi is not None and p.hi - x < max(2 * s, tiny)):
            warnings.append(f"{k} = {x:.4g} is at or within 2 sigma of a bound: its error is not meaningful")
    for pc in model.params.constraints.values():
        if pc.mode == "bounds":
            i, j = model.params._pair_index(pc)
            w = float(jnp.hypot(u[i], u[j]) / jnp.sqrt(1.0 + u[i] ** 2 + u[j] ** 2))
            if w > 0.95:
                warnings.append(f"{pc.component} {pc.keys} is at {w:.0%} of its bound ({pc.description})")
    c2, dof = model.chi2_dof(u)
    return LaplaceErrors(names, best, sig, cov, corr, float(np.sqrt(max(c2 / dof, 1.0))), warnings, H, derived)
