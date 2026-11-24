"""M12d: fitting Sersics with Fourier modes (m = 1, 3, 4).

Synthetic lensed setup of tests/test_fit.py (linear lens, mu = 2.2, toy PSF,
high S/N), with a larger source (R_e = 0.12", ~4 px) so that the modes are
measurable. Needs fourier_flux_factor (M12b): the unit-flux templates use it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import fit, laplace_errors
from forwardgal.model import ForwardModel
from forwardgal.psf import make_convolver
from test_fit import NOISE, SKY, toy_psf
from test_render_lensed import N, PIX, grid_and_centre

SHAPE = dict(R_e=0.12, n=1.0, q=0.7, pa_deg=30.0)
MODES = {1: (0.12, 40.0), 3: (0.06, 100.0), 4: (0.05, 10.0)}  # GALFIT (a_m, phi_m); phi in [0, 360/m)
FLUX = 300.0


def make_model(params, data, conv=None):
    grid, *_ = grid_and_centre(0.3, 0.2)
    rows, cols = np.nonzero(np.ones((N, N), bool))
    conv = conv or make_convolver(toy_psf(), (N, N))
    return ForwardModel(params, grid, conv, data, np.full((N, N), NOISE), np.ones((N, N), bool),
                        (rows, cols), PIX)


def test_unit_flux_template_with_modes_sums_to_one():
    """Image-plane component, no PSF: the template (pixel-sum units) sums to 1,
    i.e. amplitude = total flux of the PERTURBED profile (GALFIT convention)."""
    grid, *_ = grid_and_centre(0.3, 0.2)
    tx, ty = grid.theta_sub_x[20, 20].mean(), grid.theta_sub_y[20, 20].mean()
    c = sersic_component("fg", "image", float(tx), float(ty), R_e=0.06, n=1.0, q=0.7, pa_deg=30.0, modes=MODES)
    params = ModelParams([c])
    model = make_model(params, np.zeros((N, N)), conv=lambda img: img)
    assert float(model.templates(params.start())[0].sum()) == pytest.approx(1.0, rel=1e-3)


@pytest.fixture(scope="module")
def fitted():
    _, _, _, bx, by = grid_and_centre(0.3, 0.2)
    truth = ModelParams([sersic_component("host", "source", bx, by, **SHAPE, modes=MODES), sky_component()])
    tm = make_model(truth, np.zeros((N, N)))
    clean = np.asarray(jnp.tensordot(jnp.array([FLUX, SKY]), tm.templates(truth.start()), axes=1))
    data = clean + np.random.default_rng(5).normal(0, NOISE, clean.shape)
    start = ModelParams([sersic_component("host", "source", bx + 0.01, by - 0.01, R_e=0.15, n=1.5, q=0.8,
                                          pa_deg=40.0, modes=[1, 3, 4]), sky_component()])
    model = make_model(start, data)
    return model, fit(model, n_starts=1), (bx, by)


def test_recovers_fourier_modes(fitted):
    model, res, (bx, by) = fitted
    v = res.values["host"]
    for m, (a, phi) in MODES.items():
        assert float(v[f"a{m}"]) == pytest.approx(a, abs=0.01), m
        # phase error scales like 1 / (m a): compare on the circle of period 360/m
        d = (float(v[f"phi{m}"]) - phi + 180.0 / m) % (360.0 / m) - 180.0 / m
        assert abs(d) < 3.0, (m, float(v[f"phi{m}"]))
    assert float(v["R_e"]) == pytest.approx(SHAPE["R_e"], rel=0.03)
    assert res.amplitudes["host"] == pytest.approx(FLUX, rel=0.02)
    c2, dof = model.chi2_dof(jnp.asarray(res.u))
    assert 0.85 < c2 / dof < 1.15


def test_laplace_reports_derived_mode_amplitudes_and_phases(fitted):
    model, res, _ = fitted
    lap = laplace_errors(model, res)
    names = [d[0] for d in lap.derived]
    for m in MODES:
        assert f"host.a{m} (derived)" in names and f"host.phi{m} (derived)" in names
    a1 = dict((d[0], d) for d in lap.derived)["host.a1 (derived)"]
    assert 0 < a1[2] < 0.01  # a small, finite error at this S/N
