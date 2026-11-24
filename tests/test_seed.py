"""M14: seeding a fit (band B) from another fit's best values, with ranges."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import FourierBound, ModelParams, PositionConstraint, sersic_component, sky_component
from forwardgal.fit import _with_derived
from forwardgal.seed import seed_config, seed_params
from forwardgal.serialize import model_from_dict, model_to_dict

RANGES = {"*": {"position": 0.02, "R_e": {"rel": 0.2}, "n": 0.5, "q": 0.05, "pa_deg": 10.0}, "disk": {"n": 1.0}}


def base_and_best():
    base = ModelParams([
        sersic_component("bulge", "source", -5.73, 0.22, R_e=0.05, n=4.0, q=0.8, pa_deg=30.0),
        sersic_component("disk", "source", -5.75, 0.15, R_e=0.3, n=1.0, q=0.6, pa_deg=20.0, modes=[1], bending=[2]),
        sky_component(),
    ])
    u = base.start() + 0.02  # pretend best fit, a bit away from the start
    best = {c: {k: float(v) for k, v in d.items()} for c, d in _with_derived(base, base.unpack(u)).items()}
    return base, best


def test_ranges_set_bounds_around_the_best_fit():
    base, best = base_and_best()
    p = seed_params(base, best, RANGES)
    b, d = p.by_name["bulge"], p.by_name["disk"]
    rb = best["bulge"]["R_e"]
    assert (b.params["R_e"].lo, b.params["R_e"].hi) == pytest.approx((0.8 * rb, 1.2 * rb))
    nb = best["bulge"]["n"]
    assert (b.params["n"].lo, b.params["n"].hi) == pytest.approx((nb - 0.5, nb + 0.5))
    nd = best["disk"]["n"]  # disk's own range (1.0) overrides "*", intersected with n > 0.5
    assert (d.params["n"].lo, d.params["n"].hi) == pytest.approx((max(0.5, nd - 1.0), nd + 1.0))
    # (e1, e2) -> (q, PA) with the ranges; the |e| bound is dropped
    assert "e1" not in b.params and "q" in b.params
    q, pa = best["bulge"]["q"], best["bulge"]["pa_deg"]
    assert (b.params["q"].lo, b.params["q"].hi) == pytest.approx((q - 0.05, q + 0.05))
    assert (b.params["pa_deg"].lo, b.params["pa_deg"].hi) == pytest.approx((pa - 10, pa + 10))
    assert not any(isinstance(x, PositionConstraint) and x.keys == ("e1", "e2") for x in b.constraints)
    # modes and bends without a range: free, the Fourier bound kept
    assert d.params["c1"].lo is None and d.params["b2"].lo is None
    assert any(isinstance(x, FourierBound) for x in d.constraints)
    # centres within 0.02" of the best centres
    pos = {k[0]: x for k, x in p.constraints.items() if x.keys == ("x0", "y0")}
    assert pos["bulge"].radius == 0.02
    assert pos["bulge"].ref == pytest.approx((best["bulge"]["x0"], best["bulge"]["y0"]))


def test_seeded_start_equals_the_best_fit():
    base, best = base_and_best()
    p = seed_params(base, best, RANGES)
    v = p.unpack(p.start())
    for c in ("bulge", "disk"):
        for k in ("x0", "y0", "R_e", "n"):
            assert float(v[c][k]) == pytest.approx(best[c][k], rel=1e-9, abs=1e-12), (c, k)
    assert float(v["bulge"]["q"]) == pytest.approx(best["bulge"]["q"])
    assert float(v["disk"]["b2"]) == pytest.approx(best["disk"]["b2"])


def test_no_ranges_means_free_with_usual_bounds():
    base, best = base_and_best()
    p = seed_params(base, best)
    b = p.by_name["bulge"]
    assert "e1" in b.params and (b.params["n"].lo, b.params["n"].hi) == (0.5, 8.0)
    assert b.params["n"].value == pytest.approx(best["bulge"]["n"])
    assert not [x for x in p.constraints.values() if x.keys == ("x0", "y0")]


def test_typo_in_a_component_range_is_an_error_but_star_keys_may_not_apply():
    base, best = base_and_best()
    with pytest.raises(ValueError, match="Re"):
        seed_params(base, best, {"disk": {"Re": 0.1}})
    with pytest.raises(ValueError, match="unknown component"):
        seed_params(base, best, {"dsik": {"n": 0.1}})
    seed_params(base, best, {"*": {"n": 0.5}})  # the sky has no n: ignored


def test_best_value_on_a_bound_is_nudged_inside():
    base, best = base_and_best()
    best["disk"]["n"] = 0.5
    p = seed_params(base, best, {"disk": {"n": 0.3}})
    n = p.by_name["disk"].params["n"]
    assert n.lo == 0.5 and n.value > 0.5 and n.hi == pytest.approx(0.8)


def test_seeded_model_serializes():
    base, best = base_and_best()
    p = seed_params(base, best, RANGES)
    q = model_from_dict(model_to_dict(p))
    np.testing.assert_array_equal(np.asarray(q.start()), np.asarray(p.start()))


def test_seed_config_from_a_fit_folder(tmp_path):
    from forwardgal.fit import fit
    from forwardgal.history import new_fit_dir, save_fit
    from test_fit import TRUTH, setup

    model, _ = setup(dict(dx=0.005, **TRUTH))
    res = fit(model, n_starts=1, max_iter=5)
    f = tmp_path / "image.fits"
    f.write_bytes(b"x")
    cfg = {"label": "bandA", "data": {"image": "A"}, "model": model_to_dict(model.params)}
    folder = save_fit(new_fit_dir(tmp_path, "bandA"), cfg, model, res, {"image": f})
    new = seed_config(folder, {"host": {"position": 0.01, "n": 0.3}}, {"label": "bandB", "data": {"image": "B"}})
    assert new["label"] == "bandB" and new["data"] == {"image": "B"}
    assert new["seed"]["from_fit"] == str(folder.resolve())
    p = model_from_dict(new["model"])
    assert p.by_name["host"].params["n"].value == pytest.approx(float(res.values["host"]["n"]))


# --- real data: F150W seeded from the F115W example fit ---------------------------

from test_config import needs_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F115W_FIT = ROOT / "examples" / "fits" / "0001_1A-bulge-disk"


@needs_data
@pytest.mark.skipif(not F115W_FIT.exists(), reason="run python -m forwardgal.run examples/1A_bulge_disk.yaml first")
def test_f150w_seeded_from_f115w_builds(tmp_path):
    from forwardgal.config import build

    f150 = str(ROOT / "data" / "jw06675-o001_t002_nircam_clear-f150w_i2d.fits")
    psf150 = str(ROOT / "data" / "rotated_jw06675-o001_t002_nircam_clear-f150w-QSOblueA_PSF_i2dtrim.fits")
    import copy

    from forwardgal.serialize import read_yaml

    a = read_yaml(F115W_FIT / "model.yaml")
    data = copy.deepcopy(a["data"])
    data["image"]["file"] = data["sigma"]["file"] = f150
    # the F115W quasar position and the F150W PSF centroid differ by 0.21 px: trust the
    # PSF (max_snap None; user, 2026-11-16)
    quasar = dict(a["quasar"], max_snap=None)
    cfg = seed_config(F115W_FIT, RANGES, {"label": "1A-F150W-seeded", "data": data, "quasar": quasar,
                                          "psf": {"file": psf150, "ext": 0, "reference": "centroid"}})
    s = build(cfg, base_dir=tmp_path)
    assert s.img.zeropoint is not None and np.isfinite(float(s.model.loss(s.model.params.start())))
    assert "q" in s.model.params.by_name["disk"].params  # switched to (q, PA) with the ranges
    assert 0.1 < s.points["QSO 1A"].snap < 0.3  # measured 0.21 px


# --- G5: band names and seeding into another band's set-up ---------------------------


def test_band_name(tmp_path):
    from astropy.io import fits

    from forwardgal.config import band_name

    assert band_name({"band": "F150W"}) == "F150W"
    fits.PrimaryHDU(header=fits.Header([("FILTER", "F444W")])).writeto(tmp_path / "img.fits")
    assert band_name({"data": {"image": {"file": "img.fits"}}}, tmp_path) == "F444W"
    fits.PrimaryHDU().writeto(tmp_path / "plain.fits")
    assert band_name({"data": {"image": {"file": "plain.fits"}}}, tmp_path) == "plain"
    assert band_name({"label": "x"}) == "x"


@needs_data
@pytest.mark.skipif(not F115W_FIT.exists(), reason="run python -m forwardgal.run examples/1A_bulge_disk.yaml first")
def test_seed_into_another_bands_setup(tmp_path):
    """Band B (F150W) keeps its whole set-up and its quasar settings; the shapes come from band A's fit."""
    import copy

    from forwardgal.config import band_name, build
    from forwardgal.seed import DEFAULT_RANGES, seed_into
    from forwardgal.serialize import read_yaml

    b = copy.deepcopy(read_yaml(ROOT / "examples" / "1A_bulge_disk.yaml"))
    b["data"]["image"]["file"] = b["data"]["sigma"]["file"] = "jw06675-o001_t002_nircam_clear-f150w_i2d.fits"
    b["psf"]["file"] = "rotated_jw06675-o001_t002_nircam_clear-f150w-QSOblueA_PSF_i2dtrim.fits"
    b["label"] = "1A-F150W"
    from forwardgal.config import upgrade_config

    b = upgrade_config(b)  # B's quasar gets its own max_snap (trust B's PSF, as in M14)
    next(c for c in b["model"]["components"] if c["kind"] == "point")["max_snap"] = None
    cfg = seed_into(F115W_FIT, b, ROOT / "data", DEFAULT_RANGES)
    assert cfg["label"] == "1A-F150W" and cfg["data"]["image"]["file"].endswith("f150w_i2d.fits")
    assert Path(cfg["psf"]["file"]).is_absolute() and "f150w" in cfg["psf"]["file"]
    assert band_name(cfg) == "F150W"
    qb = next(c for c in cfg["model"]["components"] if c["kind"] == "point")
    assert qb["max_snap"] is None  # B's own
    best = read_yaml(F115W_FIT / "result.yaml")["values"]
    disk = next(c for c in cfg["model"]["components"] if c["name"] == "disk")
    assert disk["params"]["R_e"]["value"] == pytest.approx(best["disk"]["R_e"], rel=1e-6)
    assert disk["params"]["R_e"]["hi"] == pytest.approx(min(2.0, 1.2 * best["disk"]["R_e"]), rel=1e-6)
    assert not {"mag", "flux"} & set(disk["params"]) and cfg["seed"]["from_fit"] == str(F115W_FIT.resolve())
    s = build(cfg, base_dir=tmp_path)  # builds in band B (fluxes from one linear solve)
    assert np.isfinite(float(s.model.loss(s.model.params.start())))
