"""NUTS on the small synthetic lensed setup (tests/test_fit.py), high S/N.

Checks: chains agree (R-hat), no divergences, the posterior contains the
truth, and NUTS uncertainties agree with the Laplace ones (the posterior is
close to Gaussian at this S/N, so they should).
"""

import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.fit import fit, laplace_errors
from forwardgal.sample import nuts
from test_fit import FLUX, SKY, TRUTH, setup


@pytest.fixture(scope="module")
def posterior():
    model, (bx, by) = setup(dict(dx=0.0, **TRUTH))
    res = fit(model, n_starts=1)
    lap = laplace_errors(model, res)
    post = nuts(model, res, num_warmup=200, num_samples=300, num_chains=2, seed=1, progress=False)
    return post, lap, (bx, by)


def test_chains_converge_without_divergences(posterior):
    post, _, _ = posterior
    assert post.divergences == 0
    assert np.all(post.r_hat < 1.05), dict(zip(post.names, post.r_hat))
    assert np.all(post.n_eff > 50)


def test_truth_inside_posterior(posterior):
    post, _, (bx, by) = posterior
    truth = {"host.x0": bx, "host.y0": by, "host.R_e": TRUTH["R_e"], "host.n": TRUTH["n"],
             "host.q": TRUTH["q"], "amp:host": FLUX, "amp:sky": SKY}
    for name, t in truth.items():
        x = post.column(name)
        z = abs(np.median(x) - t) / np.std(x)
        assert z < 4.0, (name, np.median(x), t, np.std(x))


def test_nuts_errors_agree_with_laplace(posterior):
    post, lap, _ = posterior
    for name in ("host.x0", "host.R_e", "host.n", "amp:host", "amp:sky"):
        s_nuts = np.std(post.column(name))
        s_lap = lap.sigma[lap.names.index(name)]
        assert s_nuts == pytest.approx(s_lap, rel=0.35), (name, s_nuts, s_lap)


def test_corner_plot_runs(posterior, tmp_path):
    post, lap, _ = posterior
    post.corner(keys=["host.R_e", "host.n", "amp:host"], laplace=lap, path=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()
    assert "R-hat" in post.describe(zeropoint=28.087)
