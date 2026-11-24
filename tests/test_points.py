"""Quasar as a point component, step 1a: the YAML forms (dicts only, no images).

A point may give ra/dec (converted to locked x0, y0 by config.build) or x0/y0;
old files with a ``quasar:`` section are upgraded to a point component.
"""

import copy

import pytest

import forwardgal  # noqa: F401
from forwardgal.config import resolve_points, restore_radec, upgrade_config
from forwardgal.edit import check
from forwardgal.serialize import component_from_dict, component_to_dict, model_from_dict

RA0, DEC0 = 178.330251, 7.932518  # lens-model reference


def fake_sky_to_theta(ra, dec):
    """A stand-in for DeflectionField.sky_to_theta (flat sky, +x east = +RA): enough for dict tests."""
    return (ra - RA0) * 3600.0, (dec - DEC0) * 3600.0


def old_config():
    return {"label": "x",
            "quasar": {"name": "QSO 1A", "mode": "image_fixed", "ra": 178.3352306, "dec": 7.9335254,
                       "max_snap": 0.1},
            "model": {"components": [
                {"name": "disk", "kind": "sersic", "plane": "source",
                 "params": {"x0": {"value": -5.7}, "y0": {"value": 0.18}, "R_e": {"value": 0.25, "lo": 0.01},
                            "n": {"value": 1.0, "lo": 0.5, "hi": 8.0}, "q": {"value": 0.6, "lo": 0.05, "hi": 1.0},
                            "pa_deg": {"value": 30.0}}},
                {"name": "sky", "kind": "sky", "plane": "image", "params": {}}],
                "ties": {}, "constraints": []}}


def test_upgrade_moves_the_quasar_into_the_components_last():
    old = old_config()
    new = upgrade_config(old)
    assert "quasar" not in new and "quasar" in old  # input untouched
    q = new["model"]["components"][-1]
    assert q == {"name": "QSO 1A", "kind": "point", "plane": "image", "ra": 178.3352306, "dec": 7.9335254,
                 "max_snap": 0.1, "params": {}}
    assert [c["name"] for c in new["model"]["components"]] == ["disk", "sky", "QSO 1A"]


def test_upgrade_without_a_quasar_is_a_copy():
    cfg = old_config()
    del cfg["quasar"]
    assert upgrade_config(cfg) == cfg


def test_upgrade_refuses_other_modes_and_name_clashes():
    cfg = old_config()
    cfg["quasar"]["mode"] = "free"
    with pytest.raises(NotImplementedError):
        upgrade_config(cfg)
    cfg = old_config()
    cfg["quasar"]["name"] = "disk"
    with pytest.raises(ValueError, match="also a model component"):
        upgrade_config(cfg)


def test_resolve_gives_a_locked_point_the_fitter_accepts():
    spec = upgrade_config(old_config())["model"]
    with pytest.raises(ValueError, match="converted to x0, y0 by config.build"):
        model_from_dict(spec)  # unresolved ra/dec: clear message
    resolved, record = resolve_points(spec, fake_sky_to_theta)
    p = resolved["components"][-1]["params"]
    tx, ty = fake_sky_to_theta(178.3352306, 7.9335254)
    assert p == {"x0": {"value": tx, "free": False}, "y0": {"value": ty, "free": False}}
    assert record == {"QSO 1A": (178.3352306, 7.9335254, tx, ty)}
    params = check(resolved)
    assert params.size == 6  # the disk only: the point adds no free parameters
    assert params.by_name["QSO 1A"].max_snap == 0.1


@pytest.mark.parametrize("change, words", [
    (lambda c: c.pop("dec"), "both ra and dec"),
    (lambda c: c.update(plane="source"), "plane must be 'image'"),
    (lambda c: c["params"].update(x0={"value": 1.0}), "not both"),
    (lambda c: c.update(kind="sersic"), "only point components"),
])
def test_resolve_refuses_ambiguous_points(change, words):
    spec = upgrade_config(old_config())["model"]
    change(spec["components"][-1])
    with pytest.raises(ValueError, match=words):
        resolve_points(spec, fake_sky_to_theta)


def test_saving_restores_ra_dec_only_for_unmoved_points():
    spec = upgrade_config(old_config())["model"]
    resolved, record = resolve_points(spec, fake_sky_to_theta)
    assert restore_radec(resolved, record) == spec  # round trip: the file as written
    moved = copy.deepcopy(resolved)
    moved["components"][-1]["params"]["x0"]["value"] += 0.01
    out = restore_radec(moved, record)
    assert "ra" not in out["components"][-1] and out["components"][-1]["params"]["x0"]["free"] is False
    unlocked = copy.deepcopy(resolved)
    del unlocked["components"][-1]["params"]["y0"]["free"]
    assert "ra" not in restore_radec(unlocked, record)["components"][-1]


def test_max_snap_round_trips_including_no_limit():
    for snap in (0.1, None):
        d = {"name": "QSO", "kind": "point", "plane": "image", "max_snap": snap,
             "params": {"x0": {"value": 1.0, "free": False}, "y0": {"value": 2.0, "free": False}}}
        c = component_from_dict(d)
        assert c.max_snap == snap and component_to_dict(c) == d


# --- step 1b: the point template, drawn with the point's own PSF stamp -------------

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from forwardgal.quasar import place_point, psf_reference  # noqa: E402
from test_config import needs_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gaussian_stamp(k=15, sigma=1.5, offset=(0.0, 0.0)):
    """Unit-sum Gaussian stamp; its peak is offset (rows, cols) from the GALFIT centre."""
    yy, xx = np.indices((k, k))
    g = np.exp(-((yy - k // 2 - offset[0]) ** 2 + (xx - k // 2 - offset[1]) ** 2) / (2 * sigma**2))
    return g / g.sum()


def test_psf_reference_options():
    k = gaussian_stamp(offset=(0.3, -0.2))
    r, c = psf_reference(k, "centroid")
    assert (r, c) == pytest.approx((7.3, 6.8), abs=1e-4)  # the stamp edge cuts the Gaussian: ~4e-6 px off
    assert psf_reference(k, "centre") == (7.0, 7.0)
    assert psf_reference(k, (7.25, 6.75)) == (7.25, 6.75)
    with pytest.raises(ValueError):
        psf_reference(k, "peak")


def test_a_snapped_point_is_an_exact_copy_of_its_stamp():
    """Stamp made for a point at (40.32, 30.81) on the grid; catalogue position 0.03 px off."""
    k = gaussian_stamp(offset=(0.32, -0.19))  # centroid 7.32, 6.81 in the stamp
    pp = place_point((40.35, 30.79), k, (80, 60))
    assert pp.rc == pytest.approx((40.32, 30.81), abs=1e-4)  # moved onto the stamp's pixel grid
    assert pp.snap == pytest.approx(np.hypot(0.03, 0.02), abs=1e-4)  # (centroid ~4e-6 px off 7.32, 6.81)
    np.testing.assert_allclose(pp.template[33:48, 24:39], k, atol=1e-12)  # copied, not resampled
    assert pp.template.sum() == pytest.approx(1.0, abs=1e-9)


def test_each_point_can_have_its_own_psf():
    narrow, wide = gaussian_stamp(sigma=1.0), gaussian_stamp(sigma=2.5)
    a = place_point((40.0, 30.0), narrow, (80, 60), reference="centre")
    b = place_point((40.0, 30.0), wide, (80, 60), reference="centre")
    assert a.template.max() > 3 * b.template.max()  # same place, different stamps
    assert a.template.sum() == pytest.approx(b.template.sum(), abs=1e-9)  # both unit flux


def test_snapping_too_far_is_refused():
    with pytest.raises(ValueError, match="not seem to be made for this position"):
        place_point((40.4, 30.0), gaussian_stamp(), (80, 60), reference="centre")
    assert place_point((40.4, 30.0), gaussian_stamp(), (80, 60), reference="centre",
                       max_snap=None).snap == pytest.approx(0.4)  # "trust the PSF"


@needs_data
def test_place_point_from_theta_reproduces_todays_1A_template():
    """theta of QSO 1A -> exact pixel chain -> place_point == the template build() makes now."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import build
    from forwardgal.psf import load_psf
    from forwardgal.quasar import theta_to_grid

    cfg = example_config()
    s = build(cfg)
    qc = cfg["quasar"]
    r0, _, c0, _ = s.region.box
    tx, ty = s.field.sky_to_theta(qc["ra"], qc["dec"])  # what resolve_points stores as x0, y0
    rc = theta_to_grid(s.img.wcs, s.field, (r0, c0))(float(tx), float(ty))
    kernel = load_psf(cfg["psf"]["file"], ext=cfg["psf"].get("ext", 0)).kernel
    pp = place_point(rc, kernel, s.region.box_shape, reference=cfg["psf"].get("reference", "centroid"),
                     max_snap=qc.get("max_snap", 0.1))
    # build() now goes the theta route too, so this checks place_point against build's point template;
    # the comparison with the old way (template straight from RA/Dec, as a fixed template, chi2 bit for
    # bit) is test_config.test_yaml_setup_reproduces_python_setup_chi2_exactly.
    today = s.model.point_templates[qc["name"]]
    np.testing.assert_array_equal(pp.template, today)  # bit for bit
    assert pp.snap == pytest.approx(s.points[qc["name"]].snap, abs=1e-12)


# --- step 1c: build() with point components ------------------------------------------


@pytest.fixture(scope="module")
def built_1A():
    """The 1A example (an old-style quasar: section) built once for the 1c tests."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import build

    cfg = example_config()
    return cfg, build(cfg)


@needs_data
def test_build_turns_the_quasar_section_into_a_locked_point(built_1A):
    cfg, s = built_1A
    assert "quasar" in cfg and "quasar" not in s.cfg  # input untouched; Setup holds the upgraded form
    comp = s.cfg["model"]["components"][-1]
    assert comp["name"] == "QSO 1A" and comp["kind"] == "point" and "ra" not in comp  # resolved to x0, y0
    assert s.model.names[-1] == "QSO 1A"  # last, where the fixed template used to be
    pt = s.points["QSO 1A"]
    assert pt.radec == pytest.approx((cfg["quasar"]["ra"], cfg["quasar"]["dec"]), abs=1e-10)
    assert pt.snap < 0.1 and pt.psf == str(s.files["psf"])
    assert s.radec["QSO 1A"][:2] == (cfg["quasar"]["ra"], cfg["quasar"]["dec"])
    assert s.quasar_beta == pt.beta


def _with_point(cfg, **changes):
    from forwardgal.config import upgrade_config

    new = upgrade_config(cfg)
    new["model"]["components"][-1].update(changes)
    return new


@needs_data
def test_a_point_can_name_its_own_psf(built_1A, tmp_path):
    """Its own PSF file is used (and recorded under its own role); the global PSF stays for the rest."""
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter

    from forwardgal.config import build

    cfg, s = built_1A
    u = s.model.params.start()
    # The same file named on the point: the same model, bit for bit.
    same = build(_with_point(cfg, psf={"file": cfg["psf"]["file"], "ext": 0, "reference": "centroid"}))
    assert "psf:QSO 1A" in same.files
    assert float(same.model.loss(u)) == float(s.model.loss(u))
    # A different stamp (the 1A PSF smoothed): only the point's template changes.
    stamp = fits.getdata(cfg["psf"]["file"]).astype(float)
    fits.writeto(tmp_path / "psf_smooth.fits", gaussian_filter(stamp, 1.0))
    other = build(_with_point(cfg, psf={"file": str(tmp_path / "psf_smooth.fits")}, max_snap=None))
    assert other.points["QSO 1A"].psf == str(tmp_path / "psf_smooth.fits")
    assert not np.array_equal(other.model.point_templates["QSO 1A"], s.model.point_templates["QSO 1A"])
    assert float(other.model.loss(u)) != float(s.model.loss(u))
    np.testing.assert_array_equal(other.model.templates(u)[0], s.model.templates(u)[0])  # bulge: global PSF


@needs_data
def test_with_snapping_the_reference_only_moves_the_reported_position(built_1A):
    """Snapped, the template is the stamp copied onto whole pixels; 'centroid' and 'centre' choose the
    same whole-pixel shift unless their small offset crosses a rounding boundary (not so for 1A)."""
    from forwardgal.config import build

    cfg, s = built_1A
    centre = build(_with_point(cfg, psf={"file": cfg["psf"]["file"], "reference": "centre"}, max_snap=None))
    np.testing.assert_array_equal(centre.model.point_templates["QSO 1A"], s.model.point_templates["QSO 1A"])
    assert centre.points["QSO 1A"].snap != pytest.approx(s.points["QSO 1A"].snap, abs=1e-3)


@needs_data
def test_free_points_are_refused_for_now(built_1A):
    from forwardgal.config import build

    cfg, s = built_1A
    x0, y0 = s.points["QSO 1A"].theta
    free = _with_point(cfg, params={"x0": {"value": x0}, "y0": {"value": y0, "free": False}})
    del free["model"]["components"][-1]["ra"], free["model"]["components"][-1]["dec"]
    with pytest.raises(NotImplementedError, match="locked image position"):
        build(free)


# --- step 1c-ii: seeding another band from a fit folder with point components ---------


def fake_fit_folder(tmp_path, point_psf=True):
    """A new-format fit folder (model.yaml with an RA/Dec point, result.yaml values): enough for seed_config."""
    from forwardgal.serialize import write_yaml

    cfg = upgrade_config(old_config())
    if point_psf:
        cfg["model"]["components"][-1]["psf"] = {"file": "psf_1A_F115W.fits", "reference": "centroid"}
    cfg["psf"] = {"file": "psf_F115W.fits"}
    folder = tmp_path / "0001_x"
    folder.mkdir()
    write_yaml(cfg, folder / "model.yaml")
    values = {"disk": {"x0": -5.71, "y0": 0.17, "R_e": 0.3, "n": 1.2, "q": 0.55, "pa_deg": 32.0}, "sky": {},
              "QSO 1A": {"x0": 17.75, "y0": 3.63}}
    write_yaml({"values": values}, folder / "result.yaml")
    return folder


def test_seeding_carries_points_over_with_their_radec_and_drops_their_psf(tmp_path):
    from forwardgal.seed import seed_config

    folder = fake_fit_folder(tmp_path)
    with pytest.warns(UserWarning, match="QSO 1A"):
        cfg = seed_config(folder, {"*": {"position": 0.02}}, {"psf": {"file": "psf_F150W.fits"}})
    names = [c["name"] for c in cfg["model"]["components"]]
    assert names == ["disk", "sky", "QSO 1A"]  # the point back in its place
    q = cfg["model"]["components"][-1]
    assert (q["ra"], q["dec"]) == (178.3352306, 7.9335254) and "psf" not in q  # band B's global PSF is used
    assert cfg["seed"]["point_psf_removed"] == ["QSO 1A"]
    assert cfg["model"]["components"][0]["params"]["x0"]["value"] == -5.71  # the disk was seeded
    resolved, _ = resolve_points(cfg["model"], fake_sky_to_theta)
    check(resolved)  # a valid band-B model


def test_seeding_with_band_B_point_settings(tmp_path):
    from forwardgal.seed import seed_config

    folder = fake_fit_folder(tmp_path)
    own = {"file": "psf_1A_F150W.fits", "reference": "centroid"}
    cfg = seed_config(folder, None, None, points={"QSO 1A": {"psf": own, "max_snap": None}})  # no warning
    q = cfg["model"]["components"][-1]
    assert q["psf"] == own and q["max_snap"] is None and "point_psf_removed" not in cfg["seed"]
    with pytest.raises(ValueError, match="no point component"):
        seed_config(folder, points={"QSO 1B": {"max_snap": None}})
    with pytest.raises(ValueError, match="locked"):
        seed_config(folder, {"QSO 1A": {"position": 0.1}})


def test_an_old_style_quasar_override_replaces_the_point(tmp_path):
    from forwardgal.seed import seed_config

    folder = fake_fit_folder(tmp_path, point_psf=False)
    qso = {"name": "QSO 1A", "mode": "image_fixed", "ra": 178.3352306, "dec": 7.9335254, "max_snap": None}
    cfg = seed_config(folder, None, {"quasar": qso})
    assert "quasar" not in cfg
    assert [c["name"] for c in cfg["model"]["components"]] == ["disk", "sky", "QSO 1A"]
    assert cfg["model"]["components"][-1]["max_snap"] is None
