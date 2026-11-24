"""G2b: the display rows of the components column (gui/rows.py, no Qt)."""

import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, point_component, sersic_component, sky_component
from forwardgal.edit import check, set_param, set_tie
from forwardgal.gui.rows import param_rows, summary
from forwardgal.params import Param
from forwardgal.serialize import model_to_dict

AMPS = {"bulge": 1.2, "disk": 3.4, "QSO 1A": 9.0, "sky": 0.001}


@pytest.fixture
def spec():
    comps = [sersic_component("bulge", "source", -5.72, 0.19, 0.05, 4.0, 0.9, 30.0),  # e1, e2
             sersic_component("disk", "source", -5.72, 0.18, 0.25, 1.0, 0.6, 30.0, ellipticity="qpa"),
             point_component("QSO 1A", "image", 17.75, 3.63,
                             overrides={"x0": Param(17.75, free=False), "y0": Param(3.63, free=False)}),
             sky_component()]
    return model_to_dict(ModelParams(comps))


def rows_of(spec, name):
    p = check(spec)
    vals = {k: {a: float(b) for a, b in v.items()} for k, v in p.unpack(p.start()).items()}
    return {r.key: r for r in param_rows(spec, name, vals[name], AMPS)}, vals[name]


def test_an_e1e2_sersic_is_shown_as_q_and_pa_in_the_mockup_order(spec):
    rows, _ = rows_of(spec, "bulge")
    assert [r.label for r in rows.values()] == ["β_x (″)", "β_y (″)", "R_e (″)", "n", "q", "PA (°)", "flux"]
    assert "amp" in rows  # no flux parameter in this spec: the solved amplitude, read-only
    assert rows["q"].text == "0.900" and rows["pa_deg"].text == "30.0"  # plain numbers: units in labels
    assert rows["q"].lockable and "|e| < 0.905" in rows["q"].constraint  # the joint bound; lock converts (G2e)
    assert rows["n"].constraint == "U(0.5, 8)" and rows["n"].lockable
    assert rows["x0"].text == "-5.7200"


def test_image_plane_positions_are_theta_and_locked_points_show_locked(spec):
    rows, _ = rows_of(spec, "QSO 1A")
    assert [r.label for r in rows.values()] == ["θ_x (″)", "θ_y (″)", "flux"]
    assert rows["x0"].locked and rows["x0"].text == "+17.7500"
    assert rows["amp"].text == "9" and "NNLS" in rows["amp"].constraint and not rows["amp"].lockable


def test_ties_and_locks_are_shown(spec):
    spec = set_param(set_tie(spec, "disk.x0", "bulge.x0", 0.01), "disk", "n", free=False)
    rows, vals = rows_of(spec, "disk")
    assert rows["x0"].constraint == "= bulge.x0 + 0.01" and not rows["x0"].lockable
    assert rows["x0"].text == f"{vals['x0']:+.4f}" == "-5.7100"  # the tied (resolved) value
    assert rows["n"].locked and rows["n"].lockable
    assert rows["q"].lockable  # a (q, PA) Sersic: q is an ordinary parameter


def test_sky_and_summaries(spec):
    rows, _ = rows_of(spec, "sky")
    assert list(rows) == ["amp"] and "per pixel" in rows["amp"].constraint
    comps = {c["name"]: c for c in spec["components"]}
    assert summary(comps["bulge"], {"n": 4.0}) == "n 4.00 · source"
    assert summary(comps["QSO 1A"], {}) == "point · image" and summary(comps["sky"], {}) == "solved"
    sky = dict(comps["sky"], params={"flux": {"value": -0.002}})
    assert summary(sky, {"flux": -0.002}) == "-0.002 /px"


def test_no_amplitudes_yet():
    comps = [sersic_component("disk", "source", 0.0, 0.0, 0.25, 1.0, 0.6, 30.0)]
    spec = model_to_dict(ModelParams(comps))
    p = check(spec)
    vals = {k: float(v) for k, v in p.unpack(p.start())["disk"].items()}
    assert param_rows(spec, "disk", vals)[-1].text == "-"


def test_what_is_editable(spec):
    rows, _ = rows_of(set_tie(spec, "disk.x0", "bulge.x0", 0.0), "disk")
    assert not rows["x0"].editable and rows["y0"].editable and not rows["amp"].editable
    rows, _ = rows_of(spec, "bulge")
    assert rows["q"].editable and rows["pa_deg"].editable  # via set_ellipse
    rows, _ = rows_of(spec, "QSO 1A")
    assert not rows["x0"].editable


# --- TODO(Erik): parse_value ------------------------------------------------------------


@pytest.mark.parametrize("text, x", [
    ("0.25", 0.25), ("+0.0310", 0.031), ("-5.7223", -5.7223), ("30.0", 30.0), (" 0.25 ", 0.25),
    ("1e-3", 0.001), ("4", 4.0),
])
def test_parse_value(text, x):
    from forwardgal.gui.rows import parse_value

    assert parse_value(text) == pytest.approx(x, rel=1e-15)


@pytest.mark.parametrize("text", ["", "   ", "abc", "nan", "inf", "-inf", "1,5", "0.2 5", "30°", '0.5"'])
def test_parse_value_refuses(text):
    from forwardgal.gui.rows import parse_value

    with pytest.raises(ValueError):
        parse_value(text)


def test_a_flux_parameter_is_an_ordinary_row(spec):
    """F4a: with a 'mag' (or 'flux') parameter there is no solved-amplitude row; the flux row is editable."""
    spec = set_param(spec, "disk", "n", value=1.0)  # (just a valid edit to start from)
    disk = next(c for c in spec["components"] if c["name"] == "disk")
    disk["params"]["mag"] = {"value": 23.5, "lo": 20.0, "hi": 30.0}
    sky = next(c for c in spec["components"] if c["name"] == "sky")
    sky["params"]["flux"] = {"value": -0.002}
    rows, _ = rows_of(spec, "disk")
    assert "amp" not in rows and rows["mag"].label == "mag (AB)" and rows["mag"].text == "23.500"
    assert rows["mag"].editable and rows["mag"].lockable and rows["mag"].constraint == "U(20, 30)"
    assert "intrinsic" in rows["mag"].tooltip  # a source-plane component
    rows, _ = rows_of(spec, "sky")
    assert list(rows) == ["flux"] and rows["flux"].text == "-0.002" and "per pixel" in rows["flux"].tooltip


def test_fourier_modes_show_as_amplitude_and_phase_and_bending_as_b():
    """G6a: (c_m, s_m) are shown as a_m and phi_m (derived, editable, not lockable: the joint bound); b_m plain."""
    import numpy as np

    from forwardgal.components import ModelParams, sersic_component
    from forwardgal.gui.rows import param_rows
    from forwardgal.serialize import model_to_dict

    c = sersic_component("disk", "source", 0.0, 0.0, 0.3, 1.0, 0.6, 30.0, modes={3: (0.1, 20.0)}, bending={2: 0.01})
    spec = model_to_dict(ModelParams([c]))
    values = {k: float(x) for k, x in ModelParams([c]).unpack(ModelParams([c]).start())["disk"].items()}
    rows = {r.key: r for r in param_rows(spec, "disk", values)}
    assert "c3" not in rows and "s3" not in rows
    assert rows["a3"].label == "a_3" and float(rows["a3"].text) == pytest.approx(0.1)
    assert rows["phi3"].label == "φ_3 (°)" and float(rows["phi3"].text) == pytest.approx(20.0)
    assert rows["a3"].editable and not rows["a3"].lockable and "Σ a_m < 0.95" in rows["a3"].constraint
    assert rows["b2"].label == "b_2 (″)" and rows["b2"].lockable
    order = [r.key for r in param_rows(spec, "disk", values)]
    assert order.index("a3") < order.index("phi3") < order.index("b2")
    assert np.isfinite(float(rows["phi3"].text))
