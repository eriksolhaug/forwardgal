import jax
import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.params import Param

KINDS = {
    "both": Param(1.5, lo=0.5, hi=8.0),
    "lower": Param(0.2, lo=0.0),
    "upper": Param(-3.0, hi=1.0),
    "none": Param(0.7),
}


@pytest.mark.parametrize("kind", KINDS)
def test_roundtrip(kind):
    p = KINDS[kind]
    lo = -1e3 if p.lo is None else p.lo
    hi = 1e3 if p.hi is None else p.hi
    for x in np.linspace(lo, hi, 9)[1:-1]:
        assert float(p.to_physical(p.to_unconstrained(x))) == pytest.approx(x, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("kind", ["both", "lower", "upper"])
def test_bounds_never_crossed(kind):
    p = KINDS[kind]
    x = np.asarray(p.to_physical(jnp.linspace(-30.0, 30.0, 601)))
    if p.lo is not None:
        assert np.all(x >= p.lo)
    if p.hi is not None:
        assert np.all(x <= p.hi)


@pytest.mark.parametrize("kind", KINDS)
def test_log_jacobian_matches_derivative(kind):
    p = KINDS[kind]
    for u in (-3.0, -0.4, 0.0, 1.7, 4.0):
        dxdu = float(jax.grad(lambda v: p.to_physical(v))(u))
        assert float(p.log_jacobian(u)) == pytest.approx(np.log(abs(dxdu)), abs=1e-10)


def test_jacobian_makes_sampling_correct():
    # A flat prior on x in (0.5, 8): with the Jacobian, the density in u maps
    # back to a flat density in x. Check by integrating exp(log_jacobian) over
    # u-bins and comparing with the x-widths of the bins.
    p = KINDS["both"]
    u = np.linspace(-8, 8, 4001)
    w = np.exp(np.asarray(p.log_jacobian(u)))  # density of x per unit u (flat prior)
    x = np.asarray(p.to_physical(u))
    mass_per_bin = 0.5 * (w[1:] + w[:-1]) * np.diff(u)
    np.testing.assert_allclose(mass_per_bin, np.diff(x), rtol=1e-5)


def test_gaussian_prior():
    p = Param(2.0, prior_mu=1.0, prior_sigma=0.5)
    assert float(p.log_prior(1.0)) == 0.0
    assert float(p.log_prior(2.0)) == pytest.approx(-2.0)
    assert float(Param(2.0).log_prior(5.0)) == 0.0


def test_validation():
    with pytest.raises(ValueError, match="below"):
        Param(1.0, lo=2.0, hi=1.0)
    with pytest.raises(ValueError, match="inside the bounds"):
        Param(3.0, lo=0.0, hi=2.0)
    with pytest.raises(ValueError, match="both"):
        Param(1.0, prior_mu=1.0)
    assert "bounds (0.5, 8)" in KINDS["both"].describe()


def test_extreme_u_is_finite_and_differentiable():
    p = KINDS["both"]
    g = jax.grad(lambda u: p.to_physical(u) + p.log_jacobian(u))(40.0)
    assert np.isfinite(float(g))


# --- reference-model helpers ---------------------------------------------------------

from forwardgal.params import near, relative  # noqa: E402


def test_near_and_relative():
    g = near(-25.0, 10.0)  # PA +- 10 deg, Gaussian
    assert (g.prior_mu, g.prior_sigma, g.lo) == (-25.0, 10.0, None)
    b = relative(0.084, 0.2, mode="bounds")  # R_e within 20%, hard
    assert b.lo == pytest.approx(0.0672) and b.hi == pytest.approx(0.1008) and b.prior_mu is None
    assert relative(-2.0, 0.1).prior_sigma == pytest.approx(0.2)  # uses |ref|
    with pytest.raises(ValueError):
        relative(0.0, 0.2)
    with pytest.raises(ValueError):
        near(1.0, 0.5, mode="box")
