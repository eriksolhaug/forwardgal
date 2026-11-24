import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import nnls

import forwardgal  # noqa: F401
from forwardgal.likelihood import (
    amplitude_covariance,
    chi2,
    design_matrix,
    profile_chi2,
    solve_amplitudes,
)

RNG = np.random.default_rng(7)


def toy_templates(shape=(30, 30), centre=(15.0, 15.0), width=3.0):
    """Two blobs + a sky template."""
    y, x = (jnp.asarray(a, float) for a in np.indices(shape))  # jnp: differentiable in centre
    g1 = jnp.exp(-((y - centre[0]) ** 2 + (x - centre[1]) ** 2) / (2 * width**2))
    g2 = jnp.exp(-((y - 10) ** 2 + (x - 21) ** 2) / (2 * 1.5**2))
    return jnp.stack([g1, g2, jnp.ones(shape)])


def test_design_matrix_shape():
    t = toy_templates()
    mask = np.zeros((30, 30), bool)
    mask[5:25, 5:25] = True
    T = design_matrix(t, mask)
    assert T.shape == (400, 3)
    np.testing.assert_array_equal(np.asarray(T[:, 2]), 1.0)


def test_recovers_amplitudes_with_correct_errors():
    # Pulls (a_fit - a_true) / sqrt(cov_kk) over many noise realizations must
    # have mean ~0 and standard deviation ~1: amplitudes AND errors are right.
    t = toy_templates()
    mask = np.ones((30, 30), bool)
    T = np.asarray(design_matrix(t, mask))
    a_true = np.array([5.0, 2.0, 0.3])
    sigma = np.full(T.shape[0], 0.5)
    err = np.sqrt(np.diag(amplitude_covariance(T, sigma)))
    pulls = []
    for _ in range(300):
        d = T @ a_true + RNG.normal(0, 0.5, T.shape[0])
        a = np.asarray(solve_amplitudes(jnp.asarray(T), jnp.asarray(d), jnp.asarray(sigma)))
        pulls.append((a - a_true) / err)
    pulls = np.array(pulls)
    np.testing.assert_allclose(pulls.mean(0), 0.0, atol=0.2)
    np.testing.assert_allclose(pulls.std(0), 1.0, atol=0.15)


def test_nnls_matches_scipy_and_allows_negative_sky():
    t = toy_templates()
    T = np.asarray(design_matrix(t, np.ones((30, 30), bool)))
    d = T @ np.array([4.0, -1.0, -0.2]) + RNG.normal(0, 0.05, T.shape[0])  # blob 2 "negative"
    sigma = np.full(T.shape[0], 0.05)
    a = np.asarray(solve_amplitudes(jnp.asarray(T), jnp.asarray(d), jnp.asarray(sigma),
                                    method="nnls", nonneg=[True, True, False]))
    assert a[1] == 0.0  # flux cannot go negative
    assert a[2] < 0  # the sky can
    # With all non-negative, identical to scipy's nnls on the weighted system.
    a_all = np.asarray(solve_amplitudes(jnp.asarray(T), jnp.asarray(d), jnp.asarray(sigma), method="nnls"))
    np.testing.assert_allclose(a_all, nnls(T / 0.05, d / 0.05)[0], atol=1e-10)


def test_ridge_shrinks_towards_zero():
    t = toy_templates()
    T = jnp.asarray(design_matrix(t, np.ones((30, 30), bool)))
    d = T @ jnp.array([5.0, 2.0, 0.3])
    sigma = jnp.full(T.shape[0], 0.5)
    a0 = np.asarray(solve_amplitudes(T, d, sigma, method="ridge", ridge=1e-12))
    a1 = np.asarray(solve_amplitudes(T, d, sigma, method="ridge", ridge=1.0))
    np.testing.assert_allclose(a0, [5.0, 2.0, 0.3], rtol=1e-6)
    assert np.linalg.norm(a1) < np.linalg.norm(a0)


@pytest.mark.parametrize("method", ["lstsq", "nnls"])
def test_profile_chi2_gradient_envelope_theorem(method):
    # d chi2_min / d p, with p moving template 1, equals finite differences of
    # the fully re-solved chi2_min.
    mask = np.ones((30, 30), bool)
    d = jnp.asarray(design_matrix(toy_templates(centre=(15.3, 14.6)), mask) @ jnp.array([5.0, 2.0, 0.3]))
    d = d + jnp.asarray(RNG.normal(0, 0.05, d.shape))
    sigma = jnp.full(d.shape, 0.05)
    nonneg = [True, True, False]

    def f(p):
        T = design_matrix(toy_templates(centre=(p, 15.0)), mask)
        return profile_chi2(T, d, sigma, method, nonneg)[0]

    g = float(jax.grad(f)(15.1))
    h = 1e-5
    fd = (float(f(15.1 + h)) - float(f(15.1 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-4)


def test_works_inside_jit():
    t = toy_templates()
    T = design_matrix(t, np.ones((30, 30), bool))
    d = T @ jnp.array([5.0, 2.0, 0.3])
    sigma = jnp.full(d.shape, 0.5)
    c2, a = jax.jit(lambda T: profile_chi2(T, d, sigma))(T)
    assert float(c2) < 1e-15
    np.testing.assert_allclose(np.asarray(a), [5.0, 2.0, 0.3], rtol=1e-8)
    assert float(chi2(T, a, d, sigma)) == pytest.approx(float(c2))


# --- M9d: amplitude priors --------------------------------------------------------

from forwardgal.likelihood import amplitude_penalty, flux_prior, magnitude_prior  # noqa: E402


def test_single_amplitude_prior_matches_formula():
    # One template t: a* = (t^T W d + mu/s^2) / (t^T W t + 1/s^2).
    rng = np.random.default_rng(11)
    t = rng.uniform(0.5, 1.5, 50)
    sig = np.full(50, 0.3)
    d = 2.0 * t + rng.normal(0, 0.3, 50)
    mu, s = 1.5, 0.1
    expected = (np.sum(t * d / sig**2) + mu / s**2) / (np.sum(t**2 / sig**2) + 1 / s**2)
    a = np.asarray(solve_amplitudes(jnp.asarray(t[:, None]), jnp.asarray(d), jnp.asarray(sig),
                                    prior_mu=[mu], prior_sigma=[s]))
    assert a[0] == pytest.approx(expected, rel=1e-10)
    cov = amplitude_covariance(t[:, None], sig, prior_sigma=[s])
    assert cov[0, 0] == pytest.approx(1 / (np.sum(t**2 / sig**2) + 1 / s**2), rel=1e-10)


def test_prior_only_on_chosen_amplitudes():
    t = toy_templates()
    T = design_matrix(t, np.ones((30, 30), bool))
    d = T @ jnp.array([5.0, 2.0, 0.3])
    sigma = jnp.full(d.shape, 0.5)
    plain = np.asarray(solve_amplitudes(T, d, sigma))
    nan_prior = np.asarray(solve_amplitudes(T, d, sigma, prior_mu=[np.nan] * 3, prior_sigma=[np.nan] * 3))
    np.testing.assert_allclose(nan_prior, plain, rtol=1e-10)
    pinned = np.asarray(solve_amplitudes(T, d, sigma, prior_mu=[np.nan, 3.0, np.nan], prior_sigma=[np.nan, 1e-6, np.nan]))
    assert pinned[1] == pytest.approx(3.0, abs=1e-5)  # a very tight prior wins


def test_nnls_with_prior():
    t = toy_templates()
    T = design_matrix(t, np.ones((30, 30), bool))
    d = T @ jnp.array([5.0, 2.0, -0.2]) + jnp.asarray(RNG.normal(0, 0.05, 900))
    sigma = jnp.full(d.shape, 0.05)
    kw = dict(prior_mu=[4.0, np.nan, np.nan], prior_sigma=[0.01, np.nan, np.nan])
    a_nn = np.asarray(solve_amplitudes(T, d, sigma, method="nnls", nonneg=[True, True, False], **kw))
    a_ls = np.asarray(solve_amplitudes(T, d, sigma, method="lstsq", **kw))
    np.testing.assert_allclose(a_nn, a_ls, atol=1e-8)  # same answer when no amplitude hits 0
    # The data (900 px, sigma 0.05) constrain a_0 about as well as the prior:
    # the result is a weighted compromise between 5.0 (data) and 4.0 (prior).
    a_free = np.asarray(solve_amplitudes(T, d, sigma, method="nnls", nonneg=[True, True, False]))
    assert 4.0 < a_nn[0] < a_free[0] and a_nn[2] < 0


def test_profile_with_prior_and_envelope_gradient():
    mask = np.ones((30, 30), bool)
    d = design_matrix(toy_templates(centre=(15.3, 14.6)), mask) @ jnp.array([5.0, 2.0, 0.3])
    d = d + jnp.asarray(RNG.normal(0, 0.05, d.shape))
    sigma = jnp.full(d.shape, 0.05)

    def f(p):
        T = design_matrix(toy_templates(centre=(p, 15.0)), mask)
        # prior mean that itself depends on p, to test those gradients too
        return profile_chi2(T, d, sigma, prior_mu=jnp.array([4.0 + 0.1 * p, jnp.nan, jnp.nan]),
                            prior_sigma=jnp.array([0.2, jnp.nan, jnp.nan]))[0]

    g = float(jax.grad(f)(15.1))
    h = 1e-5
    assert g == pytest.approx((float(f(15.1 + h)) - float(f(15.1 - h))) / (2 * h), rel=1e-4)
    c2, a = profile_chi2(design_matrix(toy_templates(), mask), d, sigma, prior_mu=[5.0, np.nan, np.nan],
                         prior_sigma=[0.1, np.nan, np.nan])
    plain = chi2(design_matrix(toy_templates(), mask), a, d, sigma)
    assert float(c2) == pytest.approx(float(plain) + float(amplitude_penalty(a, [5.0, np.nan, np.nan], [0.1, np.nan, np.nan])))


def test_flux_and_magnitude_priors():
    assert flux_prior(250.0, 0.2) == (250.0, 50.0)
    F, s = magnitude_prior(22.0, 0.1, 28.087)
    assert F == pytest.approx(10 ** (-0.4 * (22.0 - 28.087)))
    assert s / F == pytest.approx(0.4 * np.log(10) * 0.1)  # 0.0921: 0.1 mag ~ 9.2% in flux
