"""M12e spiral fits: PAUSED (user, 2026-11-16); tests commented out."""

# [M12e paused] """M12e: fitting a Sersic with GALFIT-style spiral rotation (face-on, "power").

# [M12e paused] Synthetic lensed setup of tests/test_fit_modes.py; an elongated source
# [M12e paused] (q = 0.4) whose PA winds by 120 deg between 0.03" and 0.15" (two arms).
# [M12e paused] alpha is fixed at 0 (often done in GALFIT too).
# [M12e paused] """

# [M12e paused] import jax.numpy as jnp
# [M12e paused] import numpy as np
# [M12e paused] import pytest

# [M12e paused] import forwardgal  # noqa: F401
# [M12e paused] from forwardgal.components import ModelParams, sersic_component, sky_component
# [M12e paused] from forwardgal.fit import fit, laplace_errors
# [M12e paused] from forwardgal.params import Param
# [M12e paused] from test_fit import NOISE, SKY
# [M12e paused] from test_fit_modes import make_model
# [M12e paused] from test_render_lensed import N, grid_and_centre

# [M12e paused] SHAPE = dict(R_e=0.12, n=1.0, q=0.4, pa_deg=30.0)
# [M12e paused] SPIRAL = dict(kind="power", r_in=0.03, r_out=0.15, theta_out=120.0, alpha=0.0)
# [M12e paused] FLUX = 300.0
# [M12e paused] FIX_ALPHA = {"sp_alpha": Param(0.0, free=False)}


# [M12e paused] @pytest.fixture(scope="module")
# [M12e paused] def fitted():
    # [M12e paused] _, _, _, bx, by = grid_and_centre(0.3, 0.2)
    # [M12e paused] truth = ModelParams([sersic_component("host", "source", bx, by, **SHAPE, spiral=SPIRAL, overrides=FIX_ALPHA),
                         # [M12e paused] sky_component()])
    # [M12e paused] tm = make_model(truth, np.zeros((N, N)))
    # [M12e paused] clean = np.asarray(jnp.tensordot(jnp.array([FLUX, SKY]), tm.templates(truth.start()), axes=1))
    # [M12e paused] data = clean + np.random.default_rng(7).normal(0, NOISE, clean.shape)
    # [M12e paused] start = ModelParams([sersic_component("host", "source", bx + 0.01, by - 0.01, R_e=0.15, n=1.5, q=0.6,
                                          # [M12e paused] pa_deg=40.0, spiral=dict(kind="power", r_in=0.05, r_out=0.2,
                                                                   # [M12e paused] theta_out=60.0, alpha=0.0),
                                          # [M12e paused] overrides=FIX_ALPHA), sky_component()])
    # [M12e paused] model = make_model(start, data)
    # [M12e paused] return model, fit(model, n_starts=3, seed=2)


# [M12e paused] def test_recovers_spiral_winding(fitted):
    # [M12e paused] model, res = fitted
    # [M12e paused] v = res.values["host"]
    # [M12e paused] assert float(v["sp_theta"]) == pytest.approx(SPIRAL["theta_out"], abs=10.0)
    # [M12e paused] assert float(v["sp_r_in"]) == pytest.approx(SPIRAL["r_in"], abs=0.015)
    # [M12e paused] assert float(v["sp_r_out"]) == pytest.approx(SPIRAL["r_out"], abs=0.03)
    # [M12e paused] assert float(v["q"]) == pytest.approx(SHAPE["q"], abs=0.03)
    # [M12e paused] assert res.amplitudes["host"] == pytest.approx(FLUX, rel=0.02)
    # [M12e paused] c2, dof = model.chi2_dof(jnp.asarray(res.u))
    # [M12e paused] assert 0.85 < c2 / dof < 1.15


# [M12e paused] def test_laplace_reports_derived_r_out(fitted):
    # [M12e paused] model, res = fitted
    # [M12e paused] lap = laplace_errors(model, res)
    # [M12e paused] d = {x[0]: x for x in lap.derived}
    # [M12e paused] assert "host.sp_r_out (derived)" in d and d["host.sp_r_out (derived)"][2] > 0

