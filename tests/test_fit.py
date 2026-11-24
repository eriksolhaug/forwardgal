import jax
import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import fit
from forwardgal.likelihood import flux_prior
from forwardgal.model import ForwardModel
from forwardgal.psf import make_convolver
from test_render_lensed import N, PIX, grid_and_centre

TRUTH = dict(R_e=0.04, n=1.5, q=0.6, pa_deg=30.0)
FLUX, SKY, NOISE = 100.0, 0.05, 0.02


def toy_psf():
    y, x = np.mgrid[0:11, 0:11] - 5
    k = np.exp(-(x**2 + y**2) / (2 * 1.2**2))
    return k / k.sum()


def setup(start, amp_priors=None, seed=3):
    grid, w, field, bx, by = grid_and_centre(0.3, 0.2)  # lensed: mu = 2.2
    conv = make_convolver(toy_psf(), (N, N))
    rows, cols = np.nonzero(np.ones((N, N), bool))
    truth = ModelParams([sersic_component("host", "source", bx, by, **TRUTH), sky_component()])
    mask = np.ones((N, N), bool)
    true_model = ForwardModel(truth, grid, conv, np.zeros((N, N)), np.ones((N, N)), mask, (rows, cols), PIX)
    clean = np.asarray(jnp.tensordot(jnp.array([FLUX, SKY]), true_model.templates(truth.start()), axes=1))
    data = clean + np.random.default_rng(seed).normal(0, NOISE, clean.shape)
    params = ModelParams([sersic_component("host", "source", bx + start["dx"], by - start["dx"], start["R_e"],
                                           start["n"], start["q"], start["pa_deg"]), sky_component()])
    model = ForwardModel(params, grid, conv, data, np.full((N, N), NOISE), mask, (rows, cols), PIX,
                         amp_priors=amp_priors or {})
    return model, (bx, by)


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_recovers_truth_from_a_wrong_start(method):
    model, (bx, by) = setup(dict(dx=0.012, R_e=0.055, n=2.6, q=0.78, pa_deg=55.0))
    res = fit(model, n_starts=3, seed=1, method=method)
    v = res.values["host"]
    assert float(v["x0"]) == pytest.approx(bx, abs=2e-3)
    assert float(v["y0"]) == pytest.approx(by, abs=2e-3)
    assert float(v["R_e"]) == pytest.approx(TRUTH["R_e"], rel=0.05)
    assert float(v["n"]) == pytest.approx(TRUTH["n"], abs=0.15)
    assert float(v["q"]) == pytest.approx(TRUTH["q"], abs=0.03)
    assert float(v["pa_deg"]) % 180 == pytest.approx(TRUTH["pa_deg"], abs=4.0)
    assert res.amplitudes["host"] == pytest.approx(FLUX, rel=0.03)  # intrinsic flux
    assert res.amplitudes["sky"] == pytest.approx(SKY, abs=0.005)
    c2, dof = model.chi2_dof(jnp.asarray(res.u))
    assert 0.85 < c2 / dof < 1.15  # noise-level fit
    assert all(conv for _, _, conv in res.starts)


def test_amplitude_prior_pulls_the_flux():
    # A (wrong) flux prior pulls the fitted flux towards 80. With a 0.1% prior
    # the data (which pin the flux about as tightly) and the prior compromise;
    # with a far tighter prior the flux is pinned at 80.
    start = dict(dx=0.0, **TRUTH)
    free, _ = setup(start)
    some, _ = setup(start, amp_priors={"host": flux_prior(80.0, 1e-3)})
    pinned, _ = setup(start, amp_priors={"host": flux_prior(80.0, 1e-6)})
    a_free = fit(free, n_starts=1).amplitudes["host"]
    a_some = fit(some, n_starts=1).amplitudes["host"]
    a_pinned = fit(pinned, n_starts=1).amplitudes["host"]
    assert a_free == pytest.approx(FLUX, rel=0.03)
    assert 80.0 < a_some < a_free
    assert a_pinned == pytest.approx(80.0, rel=1e-3)


# --- Laplace errors -----------------------------------------------------------------

from forwardgal.fit import FitResult, laplace_errors  # noqa: E402
from forwardgal.likelihood import amplitude_covariance, design_matrix  # noqa: E402


def test_laplace_equals_exact_linear_covariance():
    # No free shape parameters (sky + one fixed template): the model is linear,
    # so the Laplace covariance must equal (T^T W T)^-1 exactly.
    rng = np.random.default_rng(5)
    grid, *_ = grid_and_centre(0.0, 0.0)
    y, x = np.mgrid[0:N, 0:N]
    blob = np.exp(-((x - 20) ** 2 + (y - 18) ** 2) / 20.0)
    sigma = np.full((N, N), 0.1)
    data = 3.0 * blob + 0.05 + rng.normal(0, 0.1, (N, N))
    mask = np.ones((N, N), bool)
    params = ModelParams([sky_component()])
    m = ForwardModel(params, grid, make_convolver(toy_psf(), (N, N)), data, sigma, mask,
                     np.nonzero(mask), PIX, fixed_templates={"blob": blob})
    _, a = m.solve(params.start())
    res = FitResult(u=np.zeros(0), loss=0.0, values={"sky": {}},
                    amplitudes={n: float(v) for n, v in zip(m.names, np.asarray(a))}, starts=[], seconds=0.0)
    lap = laplace_errors(m, res)
    T = np.asarray(design_matrix(m.templates(params.start()), mask))
    np.testing.assert_allclose(lap.covariance, amplitude_covariance(T, sigma[mask]), rtol=1e-8)


def test_laplace_hessian_matches_finite_differences():
    model, _ = setup(dict(dx=0.0, **TRUTH))
    res = fit(model, n_starts=1)
    lap = laplace_errors(model, res)
    assert np.all(np.isfinite(lap.sigma)) and np.all(lap.sigma > 0)
    # Second difference of the joint loss along u_0 (host x0) and u_2 (host R_e)
    u = jnp.asarray(res.u)
    a = jnp.array([res.amplitudes[n] for n in model.names])
    H = lap.hessian  # from finite differences of the exact gradient
    for i, h in [(0, 1e-4), (2, 1e-3)]:
        e = jnp.zeros(len(u)).at[i].set(h)
        f = lambda du: float(model.joint_loss(u + du, a))  # noqa: E731
        fd = (f(e) - 2 * f(0 * e) + f(-e)) / h**2
        assert H[i, i] == pytest.approx(fd, rel=1e-3)
    # correlations are bounded by 1, and the description runs
    assert np.all(np.abs(lap.correlation) <= 1 + 1e-9)
    assert "host.R_e" in lap.describe()


def test_residuals_sum_of_squares_equals_joint_loss():
    model, _ = setup(dict(dx=0.0, **TRUTH), amp_priors={"host": flux_prior(100.0, 0.1)})
    u = model.params.start() + 0.05
    _, a = model.solve(u)
    r = model.residuals(u, a)
    assert float(jnp.sum(r**2)) == pytest.approx(float(model.joint_loss(u, a)), rel=1e-10)
    assert float(jnp.sum(r**2)) == pytest.approx(float(model.loss(u)), rel=1e-10)  # a = a*(u)


def test_progress_printing_shows_parameters_and_bounds(capsys):
    from forwardgal.params import Param

    model, _ = setup(dict(dx=0.005, **TRUTH))
    fit(model, n_starts=1, max_iter=3, progress=True)
    out = capsys.readouterr().out
    assert "start 1/1  iteration    1" in out and "chi2/nu" in out and "lambda" in out
    assert "host" in out and " q " in out and " pa_deg " in out and " e1 " not in out
    assert "fluxes  host" in out
    # a fixed parameter is shown in brackets
    from forwardgal.components import ModelParams, sersic_component, sky_component

    p = ModelParams([sersic_component("host", "source", 0.0, 0.0, 0.05, 1.5, 0.6, 30.0,
                                      overrides={"n": Param(1.5, free=False)}), sky_component()])
    from forwardgal.fit import progress_printer

    model2 = type(model)(p, model.grid, model.convolve, model.data, model.sigma, model.mask, model.pixels,
                         model.pixscale)
    progress_printer(model2, 0, 1)(1, 100.0, p.start(), None, 1e-3)
    assert "n [1.5]" in capsys.readouterr().out
