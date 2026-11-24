"""F1/F2: flux parameters (spec, flux <-> magnitude in photometry.py) and the fitter using them."""

import math

import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.edit import check
from forwardgal.params import Param
from forwardgal.photometry import convert_param, flux_to_mag, mag_to_flux, sb_to_sky, sky_to_sb
from forwardgal.serialize import component_from_dict, model_to_dict

ZP, PIX = 28.087, 0.031  # NIRCam SW i2d-like zeropoint and pixel scale (arcsec)


# --- conversions ------------------------------------------------------------------


def test_magnitudes_round_trip_and_a_tenfold_flux_is_2_5_mag():
    assert flux_to_mag(1.0, ZP) == ZP  # unit flux is the zeropoint
    assert flux_to_mag(10.0, ZP) == pytest.approx(ZP - 2.5)
    for f in (1e-4, 0.3, 87.33):
        assert mag_to_flux(flux_to_mag(f, ZP), ZP) == pytest.approx(f, rel=1e-13)
    with pytest.raises(ValueError, match="positive flux"):
        flux_to_mag(0.0, ZP)


def test_sky_surface_brightness():
    """1 arcsec^2 holds 1/A_pix pixels: a level s per pixel is s / A_pix per arcsec^2."""
    s = 0.002
    mu = sky_to_sb(s, ZP, PIX)
    assert mu == pytest.approx(-2.5 * math.log10(s / PIX**2) + ZP)
    assert sb_to_sky(mu, ZP, PIX) == pytest.approx(s, rel=1e-13)
    with pytest.raises(ValueError, match="positive sky"):
        sky_to_sb(-1e-4, ZP, PIX)  # background-subtracted data: no surface brightness


def test_bounds_convert_exactly_and_swap_ends():
    p = {"value": 87.0, "lo": 10.0, "hi": 200.0}
    m, notes = convert_param(p, "mag", "sersic", ZP, PIX)
    assert m["lo"] == pytest.approx(flux_to_mag(200.0, ZP)) and m["hi"] == pytest.approx(flux_to_mag(10.0, ZP))
    assert m["lo"] < m["value"] < m["hi"] and notes == []
    back, _ = convert_param(m, "flux", "sersic", ZP, PIX)
    assert (back["value"], back["lo"], back["hi"]) == pytest.approx((87.0, 10.0, 200.0), rel=1e-12)


def test_zero_flux_bound_and_priors():
    m, notes = convert_param({"value": 5.0, "lo": 0.0, "free": False}, "mag", "sersic", ZP, PIX)
    assert m == {"value": pytest.approx(flux_to_mag(5.0, ZP)), "free": False} and "no faint limit" in notes[0]
    m, notes = convert_param({"value": 87.0, "prior_mu": 80.0, "prior_sigma": 4.0}, "mag", "sersic", ZP, PIX)
    assert m["prior_sigma"] == pytest.approx(1.0857 * 4.0 / 80.0, rel=1e-4) and "first order" in notes[0]


# --- the flux parameter of a component ----------------------------------------------


def test_a_component_may_carry_flux_or_mag_listed_last():
    disk = sersic_component("disk", "source", -5.7, 0.18, 0.25, 1.0, 0.6, 30.0,
                            overrides={"mag": Param(25.0, lo=20.0, hi=32.0)})
    assert disk.flux_key == "mag" and disk.names[-1] == "mag"
    sky = sky_component()
    assert sky.flux_key is None and sky.names == ()
    spec = model_to_dict(ModelParams([disk, sky]))
    params = check(spec)
    assert params.free_keys[-1] == "disk.mag"  # an ordinary free parameter


def test_flux_and_mag_together_are_refused():
    d = {"name": "sky", "kind": "sky", "plane": "image",
         "params": {"flux": {"value": 0.0}, "mag": {"value": 25.0}}}
    with pytest.raises(ValueError, match="not both"):
        component_from_dict(d)


def test_mag_parameters_need_a_zeropoint():
    from forwardgal.model import ForwardModel

    sky = component_from_dict({"name": "sky", "kind": "sky", "plane": "image", "params": {"mag": {"value": 25.0}}})
    with pytest.raises(ValueError, match="zeropoint"):
        ForwardModel(ModelParams([sky]), None, None, np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool),
                     (np.array([0]), np.array([0])), PIX)


# --- F2: the fitter with given fluxes ------------------------------------------------

from test_config import needs_data  # noqa: E402


@pytest.fixture(scope="module")
def setup_1A():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import build, upgrade_config

    cfg = upgrade_config(example_config())
    return cfg, build(cfg, flux_parameters=False)  # the old behaviour: fluxes solved (to compare against)


def _with_fluxes(cfg, amps, zp, pixscale, which, unit="flux"):
    """cfg with a flux parameter (at the given amplitudes) on the components in ``which``."""
    import copy

    from forwardgal.photometry import flux_to_mag, sky_to_sb

    new = copy.deepcopy(cfg)
    for c in new["model"]["components"]:
        if c["name"] in which:
            a = float(amps[c["name"]])
            if unit == "mag":
                v = sky_to_sb(a, zp, pixscale) if c["kind"] == "sky" else flux_to_mag(a, zp)
            else:
                v = a
            c["params"][unit] = {"value": v}
    return new


@needs_data
@pytest.mark.parametrize("which, unit", [
    ({"bulge", "disk", "sky", "QSO 1A"}, "flux"),  # every amplitude given: no NNLS at all
    ({"bulge", "disk", "QSO 1A"}, "mag"),  # magnitudes; the sky (here < 0) stays solved
    ({"QSO 1A"}, "flux"),  # mixed: the rest solved on data - quasar
])
def test_given_fluxes_at_the_nnls_values_reproduce_the_profiled_chi2(setup_1A, which, unit):
    from forwardgal.config import build

    cfg, s = setup_1A
    u = s.model.params.start()
    c2_old, dof_old = s.model.chi2_dof(u)
    _, a = s.model.solve(u)
    amps = dict(zip(s.model.names, [float(x) for x in a]))
    s2 = build(_with_fluxes(cfg, amps, s.img.zeropoint, s.img.pixscale, which, unit), flux_parameters=False)
    u2 = s2.model.params.start()
    assert s2.model.params.size == s.model.params.size + len(which)  # the fluxes are fit parameters now
    c2_new, dof_new = s2.model.chi2_dof(u2)
    assert c2_new == pytest.approx(c2_old, rel=1e-10)
    assert dof_new == dof_old  # one amplitude fewer to solve, one parameter more, per flux
    assert float(s2.model.loss(u2)) == pytest.approx(float(s.model.loss(u)), rel=1e-10)
    _, a2 = s2.model.solve(u2)
    assert dict(zip(s2.model.names, [float(x) for x in a2])) == pytest.approx(amps, rel=1e-8)
    g = s2.model.value_and_grad(u2)[1]
    assert np.all(np.isfinite(np.asarray(g)))  # differentiable in the flux parameters too


# --- F4a: adding flux parameters to a spec, and "Fit fluxes" -------------------------


def _spec():
    comps = [sersic_component("bulge", "source", -5.7, 0.2, 0.05, 4.0, 0.9, 30.0),
             sersic_component("disk", "source", -5.7, 0.18, 0.25, 1.0, 0.6, 30.0), sky_component()]
    return model_to_dict(ModelParams(comps))


def test_add_flux_parameters():
    from forwardgal.photometry import add_flux_parameters

    amps = {"bulge": 0.0, "disk": 300.0, "sky": -0.01}  # NNLS pinned the bulge at 0; the sky is negative
    spec, added = add_flux_parameters(_spec(), amps, ZP, PIX)
    p = {c["name"]: c["params"] for c in spec["components"]}
    assert added == ["bulge", "disk", "sky"]
    assert p["disk"]["mag"] == {"value": pytest.approx(flux_to_mag(300.0, ZP))}
    assert p["bulge"]["mag"]["value"] == pytest.approx(flux_to_mag(0.03, ZP))  # the faint floor: 1e-4 x 300
    assert p["sky"]["flux"] == {"value": -0.01}  # the sky: flux per pixel, may be negative
    check(spec)
    no_zp, _ = add_flux_parameters(_spec(), amps, None, PIX)
    q = {c["name"]: c["params"] for c in no_zp["components"]}
    assert q["disk"]["flux"] == {"value": 300.0, "lo": 0.0} and q["bulge"]["flux"]["value"] == pytest.approx(0.03)
    again, added = add_flux_parameters(spec, amps, ZP, PIX)
    assert added == [] and again == spec  # components that have one are left alone


def test_set_fluxes_writes_each_in_its_own_unit_and_respects_bounds():
    from forwardgal.photometry import add_flux_parameters, set_fluxes

    spec, _ = add_flux_parameters(_spec(), {"bulge": 10.0, "disk": 300.0, "sky": 0.001}, ZP, PIX)
    disk = next(c for c in spec["components"] if c["name"] == "disk")
    disk["params"]["mag"].update(lo=20.0, hi=21.0)  # a magnitude window
    disk["params"]["mag"]["value"] = 20.5
    new, notes = set_fluxes(spec, {"bulge": 20.0, "disk": 300.0, "sky": 0.002}, ZP, PIX)
    p = {c["name"]: c["params"] for c in new["components"]}
    assert p["bulge"]["mag"]["value"] == pytest.approx(flux_to_mag(20.0, ZP))
    assert p["sky"]["flux"]["value"] == 0.002
    m300 = flux_to_mag(300.0, ZP)  # = 21.89: fainter than the window allows
    assert m300 > 21.0 and p["disk"]["mag"]["value"] == pytest.approx(21.0) and "upper bound" in notes[0]


# --- F4b: the Settings menu's conversion of all flux parameters ---------------------


def test_convert_fluxes_both_ways_keeps_a_negative_sky_in_flux_and_refuses_ties():
    from forwardgal.edit import SpecError, set_tie
    from forwardgal.photometry import add_flux_parameters, convert_fluxes

    spec, _ = add_flux_parameters(_spec(), {"bulge": 10.0, "disk": 300.0, "sky": -0.001}, ZP, PIX)
    flux, notes = convert_fluxes(spec, "flux", ZP, PIX)
    p = {c["name"]: c["params"] for c in flux["components"]}
    assert "mag" not in p["disk"] and p["disk"]["flux"]["value"] == pytest.approx(300.0) and notes == []
    back, notes = convert_fluxes(flux, "mag", ZP, PIX)
    q = {c["name"]: c["params"] for c in back["components"]}
    assert q["disk"]["mag"]["value"] == pytest.approx(flux_to_mag(300.0, ZP))
    assert "flux" in q["sky"] and "kept in flux" in notes[0]  # the sky is < 0: no surface brightness
    with pytest.raises(SpecError, match="zeropoint"):
        convert_fluxes(flux, "mag", None, PIX)
    with pytest.raises(SpecError, match="untie"):
        convert_fluxes(set_tie(back, "bulge.mag", "disk.mag", 1.0), "flux", ZP, PIX)


# --- F5: build gives old files their flux parameters -------------------------------


@needs_data
def test_build_adds_flux_parameters_from_one_linear_solve(setup_1A):
    from forwardgal.config import build

    cfg, old = setup_1A  # old: fluxes solved (flux_parameters=False)
    new = build(cfg)
    assert new.fluxes_added == ["bulge", "disk", "sky", "QSO 1A"]
    kinds = {c["name"]: ("flux" if "flux" in c["params"] else "mag") for c in new.cfg["model"]["components"]}
    assert kinds == {"bulge": "mag", "disk": "mag", "sky": "flux", "QSO 1A": "mag"}
    assert new.model.solved_names == [] and new.model.params.size == old.model.params.size + 4
    c2_new, dof_new = new.model.chi2_dof(new.model.params.start())
    c2_old, dof_old = old.model.chi2_dof(old.model.params.start())
    assert c2_new == pytest.approx(c2_old, rel=1e-10) and dof_new == dof_old  # the same model at the start
    assert build(new.cfg).fluxes_added == []  # a file that has them is left alone
