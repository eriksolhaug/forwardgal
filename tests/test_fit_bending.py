"""M12f: fitting a Sersic with GALFIT-style bending modes (b2, b3).

Synthetic lensed setup of tests/test_fit_modes.py; an elongated source
(q = 0.4, R_e = 0.12") bent into a banana + S shape. The fit starts unbent.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.fit import fit, laplace_errors
from test_fit import NOISE, SKY
from test_fit_modes import make_model
from test_render_lensed import N, grid_and_centre

SHAPE = dict(R_e=0.12, n=1.0, q=0.4, pa_deg=30.0)
BENDS = {2: 0.03, 3: -0.015}  # arcsec: ridge shifts at |u| = R_e
FLUX = 300.0


@pytest.fixture(scope="module")
def fitted():
    _, _, _, bx, by = grid_and_centre(0.3, 0.2)
    truth = ModelParams([sersic_component("host", "source", bx, by, **SHAPE, bending=BENDS), sky_component()])
    tm = make_model(truth, np.zeros((N, N)))
    clean = np.asarray(jnp.tensordot(jnp.array([FLUX, SKY]), tm.templates(truth.start()), axes=1))
    data = clean + np.random.default_rng(11).normal(0, NOISE, clean.shape)
    start = ModelParams([sersic_component("host", "source", bx + 0.01, by - 0.01, R_e=0.15, n=1.5, q=0.6,
                                          pa_deg=40.0, bending=[2, 3]), sky_component()])
    model = make_model(start, data)
    return model, fit(model, n_starts=1)


def test_recovers_bending(fitted):
    model, res = fitted
    v = res.values["host"]
    for m, b in BENDS.items():
        assert float(v[f"b{m}"]) == pytest.approx(b, abs=0.003), m
    assert float(v["q"]) == pytest.approx(SHAPE["q"], abs=0.02)
    assert res.amplitudes["host"] == pytest.approx(FLUX, rel=0.02)
    c2, dof = model.chi2_dof(jnp.asarray(res.u))
    assert 0.85 < c2 / dof < 1.15


def test_laplace_reports_galfit_convention_bends(fitted):
    model, res = fitted
    lap = laplace_errors(model, res)
    d = {x[0]: x for x in lap.derived}
    for m in BENDS:
        assert f"host.b{m}_galfit (derived)" in d
    # start PA 40 lies inside GALFIT's fold: same convention, same numbers
    assert d["host.b2_galfit (derived)"][1] == pytest.approx(float(res.values["host"]["b2"]))
