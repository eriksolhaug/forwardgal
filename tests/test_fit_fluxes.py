"""F3: fitting, Laplace errors, NUTS and seeding with fluxes as fit parameters.

Uses the small synthetic lensed setup of test_fit (one source-plane Sersic
"host" + sky, known truth). The same data are modelled twice: with the
amplitudes solved by NNLS (as before F2), and with flux parameters.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams
from forwardgal.fit import fit, laplace_errors
from forwardgal.model import ForwardModel
from forwardgal.params import Param
from test_fit import TRUTH, setup

ZP = 25.0  # any zeropoint will do for the synthetic data


def with_fluxes(model, amps, unit="flux"):
    """The same model with a flux parameter on every component, starting at ``amps`` ({name: amplitude})."""
    comps = []
    for c in model.params.components:
        a = amps[c.name]
        if unit == "mag" and c.kind != "sky":  # the synthetic sky is ~0: keep it in flux
            comps.append(replace(c, params={**c.params, "mag": Param(-2.5 * np.log10(a) + ZP)}))
        else:
            comps.append(replace(c, params={**c.params, "flux": Param(a)}))
    m = model
    return ForwardModel(ModelParams(comps), m.grid, m.convolve, m.data, m.sigma, m.mask, m.pixels, m.pixscale,
                        settings=m.settings, zeropoint=ZP)


@pytest.fixture(scope="module")
def profiled():
    model, _ = setup(dict(dx=0.004, **TRUTH))
    u = model.params.start()
    _, a = model.solve(u)
    return model, u, dict(zip(model.names, [float(x) for x in a]))


def sigmas(lap):
    return dict(zip(lap.names, lap.sigma)), {n: s for n, _, s in lap.derived}


@pytest.mark.parametrize("unit", ["flux", "mag"])
def test_laplace_errors_are_the_same_with_flux_parameters(profiled, unit):
    """At the NNLS fluxes, a flux parameter is just a renaming of the amplitude: same Hessian, same errors."""
    model, u, amps = profiled
    old, _ = sigmas(laplace_errors(model, SimpleNamespace(u=u, amplitudes=amps)))
    new_model = with_fluxes(model, amps, unit)
    new, derived = sigmas(laplace_errors(new_model, SimpleNamespace(u=new_model.params.start(), amplitudes=amps)))
    for k in model.params.free_keys:  # the shape parameters
        assert new[k] == pytest.approx(old[k], rel=1e-4)
    host = new["host.flux"] if unit == "flux" else derived["host.flux (derived)"]
    assert host == pytest.approx(old["amp:host"], rel=1e-4)
    assert new["sky.flux"] == pytest.approx(old["amp:sky"], rel=1e-4)
    if unit == "mag":  # and the magnitude error itself: sigma_m = 1.0857 sigma_F / F
        assert new["host.mag"] == pytest.approx(1.0857 * old["amp:host"] / amps["host"], rel=1e-3)


def test_fit_with_flux_parameters_reaches_the_same_minimum(profiled):
    model, _, amps = profiled
    old = fit(model, n_starts=1)
    new_model = with_fluxes(model, amps, "mag")
    new = fit(new_model, n_starts=1)
    assert new.loss == pytest.approx(old.loss, rel=1e-6)
    for k in ("R_e", "n"):
        assert float(new.values["host"][k]) == pytest.approx(float(old.values["host"][k]), rel=1e-4)
    assert new.amplitudes["host"] == pytest.approx(old.amplitudes["host"], rel=1e-5)
    c2, dof = new_model.chi2_dof(jnp.asarray(new.u))
    c2_old, dof_old = model.chi2_dof(jnp.asarray(old.u))
    assert dof == dof_old and c2 == pytest.approx(c2_old, rel=1e-6)


def test_nuts_samples_flux_parameters(profiled):
    from forwardgal.sample import nuts

    model, _, amps = profiled
    new_model = with_fluxes(model, amps, "mag")
    res = fit(new_model, n_starts=1)
    post = nuts(new_model, res, num_warmup=100, num_samples=100, num_chains=1, seed=2, progress=False)
    assert "host.mag" in post.names and "amp:host" in post.names and "amp:sky" in post.names
    m = post.samples[:, post.names.index("host.mag")]
    f = post.samples[:, post.names.index("amp:host")]
    assert np.all(np.isfinite(m)) and np.allclose(f, 10 ** (-0.4 * (m - ZP)), rtol=1e-10)  # amp from the mag
    assert post.divergences == 0


def test_seeding_resets_fluxes(tmp_path):
    from forwardgal.seed import seed_config
    from test_points import fake_fit_folder

    folder = fake_fit_folder(tmp_path, point_psf=False)
    from forwardgal.serialize import read_yaml, write_yaml

    cfg = read_yaml(folder / "model.yaml")
    cfg["model"]["components"][0]["params"]["mag"] = {"value": 24.0}  # the disk had a magnitude in band A
    write_yaml(cfg, folder / "model.yaml")
    res = read_yaml(folder / "result.yaml")
    res["values"]["disk"]["mag"] = 23.8
    write_yaml(res, folder / "result.yaml")
    new = seed_config(folder, {"*": {"position": 0.02}})
    assert "mag" not in new["model"]["components"][0]["params"] and new["seed"]["fluxes_reset"] == ["disk"]
    with pytest.raises(ValueError, match="solved afresh"):
        seed_config(folder, {"disk": {"mag": 0.3}})
