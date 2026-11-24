"""G1a: LiveModel renders a spec exactly as the fitter does, re-rendering only
what changed and never recompiling for new values; set_ellipse edits a spec."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import sersic_qpa
from forwardgal.live import LiveModel, set_ellipse
from forwardgal.serialize import model_from_dict, model_to_dict
from test_fit import TRUTH, setup


@pytest.fixture(scope="module")
def synthetic():
    model, (bx, by) = setup(dict(dx=0.004, **TRUTH))
    spec = model_to_dict(model.params)
    return model, SimpleNamespace(model=model, cfg={"model": spec}), (bx, by)


def test_render_matches_the_fitter(synthetic):
    model, s, _ = synthetic
    live = LiveModel(s)
    r = live.render()
    u = model.params.start()
    c2, dof = model.chi2_dof(u)
    assert r.chi2 == pytest.approx(c2, rel=1e-9)
    assert r.dof == dof
    np.testing.assert_allclose(r.model, np.asarray(model.model_image(u)), rtol=1e-9, atol=1e-12)
    assert set(r.amplitudes) == {"host", "sky"}


def test_only_the_edited_component_is_rerendered_and_nothing_recompiles(synthetic):
    model, s, (bx, by) = synthetic
    live = LiveModel(s)
    live.render()
    sky_before = live.template("sky")
    fn = live._compiled[("template", next(k for p, k in live._compiled if p == "template" and k[0] == "sersic"))]
    n_compiled = fn._cache_size()
    for k in range(3):  # three "drags"
        live.set_spec(set_ellipse(s.cfg["model"],"host", x0=bx + 0.002 * k, R_e=0.05 + 0.01 * k))
        r = live.render()
    assert live.template("sky") is sky_before  # the unchanged sky came from the cache
    assert fn._cache_size() == n_compiled  # new values, same compiled function
    assert np.isfinite(r.chi2)


def test_set_ellipse_in_e1e2(synthetic):
    _, s, _ = synthetic
    spec = set_ellipse(s.cfg["model"], "host", q=0.5, pa_deg=-60.0)
    host = next(c for c in spec["components"] if c["name"] == "host")
    q, pa = sersic_qpa({k: host["params"][k]["value"] for k in ("e1", "e2")})
    assert float(q) == pytest.approx(0.5) and float(pa) % 180 == pytest.approx(120.0)
    assert s.cfg["model"] is not spec  # the input spec is not changed
    model_from_dict(spec)  # still a valid model


def test_set_ellipse_clips_into_bounds():
    spec = {"components": [{"name": "c", "kind": "sersic", "plane": "source", "params": {
        "x0": {"value": 0.0}, "y0": {"value": 0.0}, "R_e": {"value": 0.1, "lo": 0.01},
        "n": {"value": 1.0}, "q": {"value": 0.7, "lo": 0.05, "hi": 1.0},
        "pa_deg": {"value": 100.0, "lo": 10.0, "hi": 190.0}}}], "ties": {}, "constraints": []}
    out = set_ellipse(spec, "c", R_e=-1.0, q=1.7, pa_deg=-30.0)["components"][0]["params"]
    assert 0.01 < out["R_e"]["value"] < 0.0101
    assert out["q"]["value"] < 1.0
    assert out["pa_deg"]["value"] == pytest.approx(150.0)  # -30 + 180: inside (10, 190)
    model_from_dict({**spec, "components": [{**spec["components"][0], "params": out}]})


def test_source_image_holds_the_flux(synthetic):
    # Summed over a fine beta grid, (surface brightness per pixel area) x
    # (cell area / pixel area) gives back the component's total flux a_k.
    _, s, _ = synthetic
    live = LiveModel(s)
    r = live.render()
    v = live.values["host"]
    h, n = 0.6, 601
    g = np.linspace(-h, h, n)
    bx, by = np.meshgrid(v["x0"] + g, v["y0"] + g)
    img = live.source_image(bx, by, r.amplitudes)
    cell = (g[1] - g[0]) ** 2 / live.pixscale**2
    assert img.sum() * cell == pytest.approx(r.amplitudes["host"], rel=0.02)


# --- real data: the 1A example -----------------------------------------------------

from test_config import needs_data  # noqa: E402


@needs_data
def test_1A_example_matches_the_fitter_and_drags_fast():
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import build

    s = build(example_config(), base_dir=Path(__file__).resolve().parents[1])
    live = LiveModel(s)
    r = live.render()
    c2, _ = s.model.chi2_dof(jnp.asarray(s.model.params.start()))
    assert r.chi2 == pytest.approx(c2, rel=1e-8)
    v = live.values["disk"]
    t = []
    for k in range(5):
        live.set_spec(set_ellipse(s.cfg["model"], "disk", x0=v["x0"] + 0.01 * k))
        t0 = time.perf_counter()
        live.render()
        t.append(time.perf_counter() - t0)
    print(f"\ndrag re-render: {1e3 * np.median(t):.1f} ms (median of 5)")
    assert np.median(t) < 0.1
