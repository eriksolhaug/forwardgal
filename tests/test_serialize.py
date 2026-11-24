"""M13a: model parameters <-> plain dictionaries."""

import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import (
    ModelParams,
    pa_window,
    point_component,
    regularize,
    sersic_component,
    sky_component,
    within,
)
from forwardgal.params import Param
from forwardgal.serialize import model_from_dict, model_to_dict, param_from_dict, param_to_dict

# --- single parameters --------------------------------------------------------

PARAMS = [
    Param(1.5),
    Param(0.3, free=False),
    Param(0.05, lo=0.01),
    Param(4.0, lo=0.5, hi=8.0),
    Param(-2.0, hi=0.0),
    Param(0.2, prior_mu=0.0, prior_sigma=0.4),
    Param(30.0, free=False, lo=-60.0, hi=120.0, prior_mu=30.0, prior_sigma=5.0),
]


@pytest.mark.parametrize("p", PARAMS)
def test_param_round_trip(p):
    assert param_from_dict(param_to_dict(p)) == p


def test_param_dict_omits_defaults():
    assert param_to_dict(Param(1.5)) == {"value": 1.5}
    assert param_to_dict(Param(0.05, lo=0.01)) == {"value": 0.05, "lo": 0.01}


def test_param_from_dict_uses_defaults():
    assert param_from_dict({"value": 2.0}) == Param(2.0)
    assert param_from_dict({"value": 2.0, "free": False}) == Param(2.0, free=False)


def test_param_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="upper"):
        param_from_dict({"value": 1.0, "upper": 2.0})


def test_param_from_dict_needs_a_value():
    with pytest.raises((ValueError, KeyError)):
        param_from_dict({"lo": 0.0})


def test_param_from_dict_still_checks_bounds():
    with pytest.raises(ValueError):
        param_from_dict({"value": 10.0, "lo": 0.5, "hi": 8.0})


# --- whole models -----------------------------------------------------------------


def rich_model():
    """Everything the model files must carry."""
    bulge = regularize(sersic_component("bulge", "source", -5.73, 0.22, R_e=0.05, n=4.0, q=0.8, pa_deg=30.0))
    disk = sersic_component("disk", "source", -5.75, 0.15, R_e=0.3, n=1.0, q=0.6, pa_deg=120.0,
                            ellipticity="qpa", overrides={"pa_deg": pa_window(120.0)},
                            modes={1: (0.05, 30.0), 4: (0.0, 0.0)}, bending={2: 0.01, 3: 0.0})
    star = point_component("star", "image", 17.2, 4.1)
    arm = sersic_component("arm", "source", -5.83, -0.12, R_e=0.17, n=1.0, q=0.25, pa_deg=171.0,
                           overrides={"n": Param(1.0, free=False)})
    return ModelParams([bulge, disk, star, arm, sky_component()],
                       ties={"arm.y0": ("disk.y0", -0.27)},
                       constraints=[within(disk, (-5.75, 0.15), 0.1, "arcsec")])


def plain(x):
    """True if x contains only YAML-friendly plain types."""
    if isinstance(x, dict):
        return all(isinstance(k, str) and plain(v) for k, v in x.items())
    if isinstance(x, list):
        return all(plain(v) for v in x)
    return type(x) in (str, float, int, bool)


def test_model_dict_is_plain_and_round_trips_exactly():
    p = rich_model()
    d = model_to_dict(p)
    assert plain(d)
    q = model_from_dict(d)
    assert model_to_dict(q) == d
    assert q.free_keys == p.free_keys
    np.testing.assert_array_equal(np.asarray(q.start()), np.asarray(p.start()))
    u = p.start() + 0.01
    vp, vq = p.unpack(u), q.unpack(u)
    for c in vp:
        for k in vp[c]:
            assert float(vq[c][k]) == float(vp[c][k]), (c, k)
    assert float(q.log_prior(u)) == float(p.log_prior(u))
    assert q.by_name["disk"].bend_pa_ref == 120.0


def test_model_from_dict_rejects_unknown_keys():
    d = model_to_dict(rich_model())
    d["component"] = []
    with pytest.raises(ValueError, match="unknown"):
        model_from_dict(d)
    d = model_to_dict(rich_model())
    d["components"][0]["plnae"] = "source"
    with pytest.raises(ValueError, match="plnae"):
        model_from_dict(d)


# --- YAML files (M13b) --------------------------------------------------------

from forwardgal.serialize import read_yaml, write_yaml  # noqa: E402


def test_yaml_round_trip_is_exact_for_awkward_floats(tmp_path):
    vals = [0.1, 1 / 3, 1e-300, -0.0, 123456789.123456789, 2.5e17, float("inf"), -float("inf")]
    write_yaml({"x": vals}, tmp_path / "f.yaml")
    back = read_yaml(tmp_path / "f.yaml")["x"]
    assert [repr(v) for v in back] == [repr(v) for v in vals]  # repr: also tells -0.0 from 0.0


def test_model_yaml_round_trip_rebuilds_the_model(tmp_path):
    p = rich_model()
    write_yaml(model_to_dict(p), tmp_path / "model.yaml", header="forwardgal model\ntest")
    text = (tmp_path / "model.yaml").read_text()
    assert text.startswith("# forwardgal model\n# test\n")
    assert text.index("bulge") < text.index("disk") < text.index("star")  # order kept
    q = model_from_dict(read_yaml(tmp_path / "model.yaml"))
    assert model_to_dict(q) == model_to_dict(p)
    np.testing.assert_array_equal(np.asarray(q.start()), np.asarray(p.start()))
