"""N2: fits without a lens (GALFIT-like): image-plane components only, theta from the cutout centre."""

import copy
import sys

import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.config import build
from forwardgal.serialize import model_to_dict
from test_config import ROOT, needs_data


def example():
    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    return example_config()


def host_model(x0, y0):
    """An image-plane Sersic 'host' at theta (x0, y0) + sky."""
    return model_to_dict(ModelParams([sersic_component("host", "image", x0, y0, R_e=0.3, n=1.5, q=0.7,
                                                       pa_deg=40.0), sky_component()]))


def no_lens_config(**data):
    cfg = example()
    del cfg["lens"], cfg["cosmology"]
    cfg["data"].update(data)
    cfg["model"] = host_model(0.05, -0.05)
    return cfg


@pytest.fixture(scope="module")
def flat():
    return build(no_lens_config(), flux_parameters=False)


@needs_data
def test_theta_is_measured_from_the_cutout_centre(flat):
    """The example cutout is centred on quasar 1A, so without a lens the quasar is at theta = 0."""
    s = flat
    assert not s.lensed and s.field.z_ref != s.field.z_ref  # NaN: no redshift
    q = s.points["QSO 1A"]
    assert q.theta == pytest.approx((0.0, 0.0), abs=1e-6)
    assert q.beta == pytest.approx(q.theta, abs=0) and q.mu == pytest.approx(1.0)
    assert "deflection_x" not in s.files


@needs_data
def test_reference_moves_the_origin():
    """reference 1" east and 2" north of the centre: the quasar is at theta = (-1, -2)."""
    ra_c, dec_c = example()["data"]["centre"]
    ref = [ra_c + 1.0 / 3600 / np.cos(np.radians(dec_c)), dec_c + 2.0 / 3600]
    s = build(no_lens_config(reference=ref), flux_parameters=False)
    assert s.points["QSO 1A"].theta == pytest.approx((-1.0, -2.0), abs=1e-4)


@needs_data
def test_same_sky_position_same_model_with_or_without_the_lens(flat):
    """Image-plane components never use the deflection: the lensed set-up with the host at the same sky
    position renders the same image (only theta's origin differs)."""
    cfg = example()
    ra, dec = flat.field.theta_to_sky(0.05, -0.05)  # where the no-lens host is
    lensed = build(cfg, flux_parameters=False)  # (just for its field: theta of that sky position)
    tx, ty = (float(t) for t in lensed.field.sky_to_theta(ra, dec))
    cfg["model"] = host_model(tx, ty)
    lensed = build(cfg, flux_parameters=False)
    assert lensed.lensed
    m1, m2 = lensed.model, flat.model
    np.testing.assert_array_equal(m1.mask, m2.mask)
    u1, u2 = m1.params.start(), m2.params.start()
    a1, a2 = m1.solve(u1)[1], m2.solve(u2)[1]
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=1e-5)  # host, QSO, sky amplitudes
    # Not bit for bit: theta lives in a tangent plane at the reference, and the two references (lens model,
    # quasar 1A) are 18" apart, so the planes are rotated by ~ (18"/206265") sin(Dec) ~ 1e-5 rad: the same
    # PA points in very slightly different sky directions. chi2 agrees to ~1e-6.
    assert float(m1.chi2_dof(u1)[0]) == pytest.approx(float(m2.chi2_dof(u2)[0]), rel=1e-5)


@needs_data
@pytest.mark.parametrize("change, words", [
    (lambda c: c["model"]["components"][0].update(plane="source"), "no source plane"),
    (lambda c: c.update(lens=example()["lens"], cosmology=example()["cosmology"]) or
     c["data"].update(reference=c["data"]["centre"]), "reference is for fits without a lens"),
])
def test_refused(change, words):
    cfg = no_lens_config()
    change(cfg)
    with pytest.raises(ValueError, match=words):
        build(cfg, flux_parameters=False)


@needs_data
def test_lens_null_is_the_same_as_no_lens_section(flat):
    cfg = no_lens_config()
    cfg["lens"] = None
    s = build(copy.deepcopy(cfg), flux_parameters=False)
    u = s.model.params.start()
    assert float(s.model.chi2_dof(u)[0]) == float(flat.model.chi2_dof(flat.model.params.start())[0])


@needs_data
def test_a_fit_without_a_lens_runs_saves_and_reloads(tmp_path):
    """The whole pipeline (fit, Laplace errors, history folder, summary plot) without a lens."""
    import jax.numpy as jnp

    from forwardgal.config import load
    from forwardgal.history import read_fit
    from forwardgal.run import run
    from forwardgal.serialize import write_yaml

    cfg = no_lens_config()
    cfg["label"] = "flat"
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 30}
    write_yaml(cfg, tmp_path / "flat.yaml")
    folder, s, res, lap = run(tmp_path / "flat.yaml", errors=True, plot=True, progress=False)
    assert (folder / "fit_summary.png").exists() and "beta" not in (folder / "log.txt").read_text()
    from astropy.io import fits

    with fits.open(folder / "imgblock.fits") as h:  # the GALFIT-style image block
        assert [x.name for x in h][1:4] == ["DATA", "MODEL", "RESIDUAL"] and h[0].header["LENSED"] is False
    assert np.all(np.isfinite(lap.sigma))
    got = read_fit(folder)
    assert "lens" not in got["config"]
    again = load(folder / "model.yaml")
    assert not again.lensed
    assert float(again.model.loss(jnp.asarray(got["arrays"]["u"]))) == got["result"]["loss"]
