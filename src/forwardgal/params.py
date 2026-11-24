"""Model parameters: bounds, priors and the unconstrained representation.

Optimizers and NUTS work on unconstrained real numbers u. A physical
parameter x with hard bounds is written as a smooth function x(u):

    both bounds (lo, hi) : x = lo + (hi - lo) * sigmoid(u),  u = log((x - lo)/(hi - x))
    lower bound only     : x = lo + softplus(u),             u = log(expm1(x - lo))
    upper bound only     : x = hi - softplus(u),             u = log(expm1(hi - x))
    no bounds            : x = u

with sigmoid(u) = 1/(1 + e^-u) and softplus(u) = log(1 + e^u), so x can never
leave its bounds and nothing jumps at the edges.

Densities: a prior p(x) becomes p(x(u)) |dx/du| in u, so a sampler working in
u must add log|dx/du| (``log_jacobian``) to the log-probability. An optimizer
looking for the best fit does not need that term.

Gaussian priors are -1/2 ((x - mu)/sigma)^2 (normalization constants
dropped). Hard bounds alone mean a flat prior between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus


def _inverse_softplus(z):
    """u with softplus(u) = z > 0, i.e. log(e^z - 1), without overflow.

    log(e^z - 1) = z + log(1 - e^-z) = z + log(-expm1(-z)): e^z never appears,
    so large z (e.g. 750) does not overflow to inf.
    """
    z = jnp.asarray(z, dtype=float)
    return z + jnp.log(-jnp.expm1(-z))


@dataclass(frozen=True)
class Param:
    """One model parameter.

    Attributes
    ----------
    value : float
        Starting value (or the fixed value if ``free`` is False).
    free : bool
        Whether the fit varies it.
    lo, hi : float or None
        Hard bounds (exclusive); None for no bound.
    prior_mu, prior_sigma : float or None
        Gaussian prior; both None for no Gaussian prior.
    """

    value: float
    free: bool = True
    lo: float | None = None
    hi: float | None = None
    prior_mu: float | None = None
    prior_sigma: float | None = None

    def __post_init__(self):
        if self.lo is not None and self.hi is not None and not self.lo < self.hi:
            raise ValueError(f"lower bound {self.lo} must be below upper bound {self.hi}")
        if not self.within_bounds(self.value):
            raise ValueError(f"value {self.value} is not inside the bounds ({self.lo}, {self.hi})")
        if (self.prior_mu is None) != (self.prior_sigma is None):
            raise ValueError("give both prior_mu and prior_sigma, or neither")
        if self.prior_sigma is not None and self.prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive")

    def within_bounds(self, x):
        return (self.lo is None or x > self.lo) and (self.hi is None or x < self.hi)

    # --- transforms (work on floats, numpy or JAX arrays) ---------------------

    def to_physical(self, u):
        """x(u)."""
        if self.lo is not None and self.hi is not None:
            return self.lo + (self.hi - self.lo) * sigmoid(u)
        if self.lo is not None:
            return self.lo + softplus(u)
        if self.hi is not None:
            return self.hi - softplus(u)
        return u

    def to_unconstrained(self, x):
        """u(x), the inverse of ``to_physical``."""
        if self.lo is not None and self.hi is not None:
            return jnp.log((x - self.lo) / (self.hi - x))
        if self.lo is not None:
            return _inverse_softplus(x - self.lo)
        if self.hi is not None:
            return _inverse_softplus(self.hi - x)
        return jnp.asarray(x, dtype=float)

    def log_jacobian(self, u):
        """log |dx/du|."""
        if self.lo is not None and self.hi is not None:
            return np.log(self.hi - self.lo) - softplus(u) - softplus(-u)
        if self.lo is not None or self.hi is not None:
            return -softplus(-u)  # log sigmoid(u) = d softplus / du
        return jnp.zeros_like(jnp.asarray(u, dtype=float))

    def log_prior(self, x):
        """Gaussian log prior (constants dropped); 0 if there is none."""
        if self.prior_mu is None:
            return jnp.zeros_like(jnp.asarray(x, dtype=float))
        return -0.5 * ((x - self.prior_mu) / self.prior_sigma) ** 2

    def describe(self):
        bits = [f"{self.value:g}", "free" if self.free else "fixed"]
        if self.lo is not None or self.hi is not None:
            bits.append(f"bounds ({'-inf' if self.lo is None else f'{self.lo:g}'}, "
                        f"{'inf' if self.hi is None else f'{self.hi:g}'})")
        if self.prior_mu is not None:
            bits.append(f"prior N({self.prior_mu:g}, {self.prior_sigma:g})")
        return ", ".join(bits)


# --- constraints relative to a reference model (e.g. the fit in another band) ---


def near(ref, width, mode="gaussian", value=None, free=True):
    """Param constrained to ref +- width (absolute width, e.g. PA +- 10 deg).

    mode "gaussian": prior N(ref, width), no bounds. mode "bounds": hard bounds
    (ref - width, ref + width), flat prior inside. ``value`` = starting value
    (default ref).
    """
    if width <= 0:
        raise ValueError("width must be positive")
    v = ref if value is None else value
    if mode == "gaussian":
        return Param(v, free=free, prior_mu=ref, prior_sigma=width)
    if mode == "bounds":
        return Param(v, free=free, lo=ref - width, hi=ref + width)
    raise ValueError(f"mode must be 'gaussian' or 'bounds', got {mode!r}")


def relative(ref, frac, mode="gaussian", value=None, free=True):
    """Param constrained to ref +- frac * |ref| (e.g. R_e within 20%: frac = 0.2)."""
    if ref == 0:
        raise ValueError("a relative constraint needs a non-zero reference value")
    return near(ref, frac * abs(ref), mode=mode, value=value, free=free)
