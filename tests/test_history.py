"""M13d: one folder per fit (writer, reader, listing, checksums, reproducibility)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.fit import fit, laplace_errors
from forwardgal.history import bestfit_config, list_fits, new_fit_dir, read_fit, save_fit
from forwardgal.serialize import model_from_dict, model_to_dict
from test_fit import TRUTH, setup


@pytest.fixture(scope="module")
def synthetic_fit():
    model, _ = setup(dict(dx=0.005, **TRUTH))
    res = fit(model, n_starts=1)
    return model, res, laplace_errors(model, res)


def test_folders_are_numbered(tmp_path):
    a = new_fit_dir(tmp_path, "bulge disk")
    b = new_fit_dir(tmp_path, "bulge disk")
    (tmp_path / "notes").mkdir()  # foreign folders are ignored
    c = new_fit_dir(tmp_path, "x/y")
    assert [a.name, b.name, c.name] == ["0001_bulge-disk", "0002_bulge-disk", "0003_x-y"]


def test_save_read_list(tmp_path, synthetic_fit):
    model, res, lap = synthetic_fit
    data = tmp_path / "image.fits"
    data.write_bytes(b"pretend FITS")
    cfg = {"label": "synthetic", "model": model_to_dict(model.params)}
    folder = save_fit(new_fit_dir(tmp_path / "fits", "synthetic"), cfg, model, res, {"image": data}, laplace=lap,
                      zeropoint=28.0)
    got = read_fit(folder)
    assert got["warnings"] == []
    r = got["result"]
    assert r["loss"] == res.loss and r["chi2_per_dof"] > 0
    assert r["laplace"]["parameters"]["host.R_e"]["sigma"] == pytest.approx(lap.sigma[lap.names.index("host.R_e")])
    assert "host.q (derived)" in r["laplace"]["derived"]
    np.testing.assert_array_equal(got["arrays"]["u"], np.asarray(res.u))
    assert got["config"] == cfg
    # the saved model reproduces the saved loss exactly at the saved u
    p = model_from_dict(got["config"]["model"])
    assert p.free_keys == model.params.free_keys
    assert float(model.loss(jnp.asarray(got["arrays"]["u"]))) == r["loss"]
    rows = list_fits(tmp_path / "fits")
    assert [x["folder"] for x in rows] == ["0001_synthetic"] and rows[0]["converged"] is True


def test_changed_input_file_is_reported(tmp_path, synthetic_fit):
    model, res, _ = synthetic_fit
    data = tmp_path / "image.fits"
    data.write_bytes(b"original")
    folder = save_fit(new_fit_dir(tmp_path, "s"), {"label": "s", "model": model_to_dict(model.params)}, model, res,
                      {"image": data})
    data.write_bytes(b"modified")
    with pytest.warns(UserWarning, match="changed"):
        got = read_fit(folder)
    assert len(got["warnings"]) == 1
    data.unlink()
    with pytest.warns(UserWarning, match="no longer exists"):
        read_fit(folder)


def test_bestfit_config_restarts_at_the_best_fit(synthetic_fit):
    model, res, _ = synthetic_fit
    cfg = {"label": "s", "model": model_to_dict(model.params)}
    new, nudged = bestfit_config(cfg, res)
    assert nudged == []
    v = model_from_dict(new["model"]).unpack(model_from_dict(new["model"]).start())["host"]
    for k in ("x0", "R_e", "n"):
        assert float(v[k]) == pytest.approx(float(res.values["host"][k]), rel=1e-9, abs=1e-12)


def test_bestfit_value_on_a_bound_is_nudged_inside(synthetic_fit):
    model, res, _ = synthetic_fit
    cfg = {"label": "s", "model": model_to_dict(model.params)}
    values = {c: dict(v) for c, v in res.values.items()}
    values["host"]["n"] = 0.5  # exactly at the lower bound, as the 1A disk
    fake = type(res)(res.u, res.loss, values, res.amplitudes, res.starts, res.seconds)
    new, nudged = bestfit_config(cfg, fake)
    assert nudged == ["host.n"]
    model_from_dict(new["model"])  # a valid model (value strictly inside)


# --- real data: a short run of the example model, reloaded -------------------------

from test_config import needs_data  # noqa: E402


@needs_data
def test_run_and_reload_reproduces_chi2(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import load
    from forwardgal.run import run
    from forwardgal.serialize import write_yaml

    cfg = example_config()
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 2}
    write_yaml(cfg, tmp_path / "model.yaml")
    folder, s, res, _ = run(tmp_path / "model.yaml", errors=False, plot=False)
    assert folder == tmp_path / "fits" / "0001_1A-bulge-disk"
    got = read_fit(folder)
    assert got["warnings"] == []
    again = load(folder / "model.yaml")
    run_disk = next(c for c in got["config"]["model"]["components"] if c["name"] == "disk")
    assert "mag" in run_disk["params"]  # saved as run: with the flux parameters build added (F5)
    assert float(again.model.loss(jnp.asarray(got["arrays"]["u"]))) == got["result"]["loss"]


# --- G3: drafts, the history table, trash -------------------------------------------


from forwardgal.serialize import write_yaml  # noqa: E402


def _spec(*kinds):
    from forwardgal.components import ModelParams, point_component, sersic_component, sky_component
    from forwardgal.serialize import model_to_dict

    make = {"sersic": lambda k: sersic_component(f"s{k}", "source", 0.0, 0.0, 0.1, 1.0, 0.8, 0.0),
            "point": lambda k: point_component(f"p{k}", "image", 1.0, 2.0), "sky": lambda k: sky_component()}
    return model_to_dict(ModelParams([make[x](k) for k, x in enumerate(kinds)]))


def test_components_text():
    from forwardgal.history import components_text

    assert components_text(_spec("sersic", "sersic", "point", "sky")) == "2 Sérsic + point + sky"
    assert components_text(_spec("sersic", "sky")) == "1 Sérsic + sky"
    spec = _spec("sersic")
    spec["components"][0]["params"]["c3"] = {"value": 0.0}
    assert components_text(spec) == "1 Sérsic + Fourier"
    assert components_text({"components": []}) == "empty"


def test_bic():
    from forwardgal.history import bic

    assert bic(1000.0, 990, 1000) == pytest.approx(1000.0 + 10 * np.log(1000))  # k = N - dof = 10
    assert bic(1000.0, 990, None) is None and bic(None, 990, 1000) is None


def test_drafts_and_fits_in_one_numbered_history(tmp_path):
    from forwardgal.history import list_history, save_draft

    hd = tmp_path / "fits"
    cfg = {"label": "1A", "model": _spec("sersic", "sky")}
    d1 = save_draft(hd, cfg, chi2=2900.0, dof=1000, n_pixels=1010)
    assert d1.name == "0001_1A" and (d1 / "model.yaml").exists() and (d1 / "draft.yaml").exists()
    # a "fit" folder (result.yaml as save_fit writes it), started from the draft
    f2 = hd / "0002_1A"
    f2.mkdir()
    write_yaml(cfg, f2 / "model.yaml")
    write_yaml({"label": "1A", "created": "2026-11-23T10:00:00", "chi2": 1100.0, "dof": 1000, "chi2_per_dof": 1.1,
                "n_pixels": 1010, "started_from": d1.name,
                "starts": [{"loss": 1.0, "iterations": 9, "converged": True}]}, f2 / "result.yaml")
    d3 = save_draft(hd, dict(cfg, label="1A-clump"), started_from=f2.name)
    (hd / "notes").mkdir()  # not a fit or draft folder: ignored
    rows = list_history(hd)
    assert [r["folder"] for r in rows] == ["0001_1A", "0002_1A", "0003_1A-clump"]
    assert [r["kind"] for r in rows] == ["draft", "fit", "draft"]
    assert [r["status"] for r in rows] == ["unfitted", "optimizer", "unfitted"]
    assert [r["started_from"] for r in rows] == [None, "0001_1A", "0002_1A"]
    assert rows[0]["chi2_per_dof"] == pytest.approx(2.9) and rows[2]["chi2_per_dof"] is None
    assert rows[1]["bic"] == pytest.approx(1100 + 10 * np.log(1010)) and rows[1]["components"] == "1 Sérsic + sky"
    assert d3.name == "0003_1A-clump"


def test_trash_moves_and_restore_brings_back(tmp_path):
    from forwardgal.history import list_history, list_trash, restore, save_draft, trash

    hd = tmp_path / "fits"
    d = save_draft(hd, {"label": "a", "model": _spec("sky")})
    trash(hd, d.name)
    assert list_history(hd) == [] and list_trash(hd) == ["0001_a"] and (hd / ".trash" / "0001_a" / "model.yaml").exists()
    restore(hd, "0001_a")
    assert [r["folder"] for r in list_history(hd)] == ["0001_a"] and list_trash(hd) == []
    with pytest.raises(FileNotFoundError):
        trash(hd, "0009_nope")
    trash(hd, "0001_a")
    d2 = save_draft(hd, {"label": "a", "model": _spec("sky")})  # numbers are never reused (trash counts)
    assert d2.name == "0002_a"
    # a folder of the same name trashed twice (e.g. one made again by hand): both are kept
    (hd / "0001_a").mkdir()
    trash(hd, "0001_a")
    assert list_trash(hd) == ["0001_a", "0001_a~1"]


def test_absolute_paths_and_the_history_dir_of_a_history_yaml(tmp_path):
    from forwardgal.config import absolute_paths
    from forwardgal.history import history_dir_for, save_draft

    cfg = {"data": {"image": {"file": "img.fits"}, "sigma": {"file": "~/err.fits"}},
           "lens": {"deflection_x": "maps/x.fits", "deflection_y": "/abs/y.fits"},
           "psf": {"file": "psf.fits"}, "history_dir": "hist",
           "model": {"components": [{"name": "Q", "kind": "point", "psf": {"file": "q.fits"}, "params": {}}]}}
    a = absolute_paths(cfg, tmp_path)
    assert a["data"]["image"]["file"] == str((tmp_path / "img.fits").resolve())
    assert a["data"]["sigma"]["file"] == str(Path("~/err.fits").expanduser())
    assert a["lens"]["deflection_x"] == str((tmp_path / "maps/x.fits").resolve())
    assert a["lens"]["deflection_y"] == "/abs/y.fits" and a["history_dir"] == str((tmp_path / "hist").resolve())
    assert a["model"]["components"][0]["psf"]["file"] == str((tmp_path / "q.fits").resolve())
    assert cfg["psf"]["file"] == "psf.fits"  # input untouched

    hd = tmp_path / "fits"
    d = save_draft(hd, {"label": "x", "model": {"components": []}})
    assert history_dir_for(d / "model.yaml", {}) == hd  # its own history, not fits/0001_x/fits
    assert history_dir_for(tmp_path / "m.yaml", {}) == tmp_path / "fits"


def test_an_old_quasar_section_counts_as_a_point(tmp_path):
    from forwardgal.history import list_history, save_draft

    cfg = {"label": "old", "model": _spec("sersic", "sky"),
           "quasar": {"name": "QSO", "mode": "image_fixed", "ra": 178.3, "dec": 7.9}}
    save_draft(tmp_path, cfg)
    assert list_history(tmp_path)[0]["components"] == "1 Sérsic + point + sky"


@needs_data
def test_run_only_some_components(tmp_path):
    """G4 "Fit selected": run(..., only=["disk"]) fits the disk alone; the bulge comes back unchanged; the
    folder's model keeps the file's own free/locked settings and result.yaml says what was fitted."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.history import list_history
    from forwardgal.run import run
    from forwardgal.serialize import read_yaml

    cfg = example_config()
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 3}
    write_yaml(cfg, tmp_path / "model.yaml")
    folder, s, res, _ = run(tmp_path / "model.yaml", errors=False, plot=False, progress=False, only=["disk"])
    assert {k.split(".")[0] for k in s.model.params.free_keys} == {"disk"}
    start = {c["name"]: c["params"] for c in cfg["model"]["components"]}
    for k in ("x0", "y0", "R_e", "n", "e1", "e2"):
        assert float(res.values["bulge"][k]) == pytest.approx(start["bulge"][k]["value"], abs=1e-12)
    assert float(res.values["disk"]["R_e"]) != start["disk"]["R_e"]["value"]  # the disk was fitted
    run_model = {c["name"]: c for c in read_yaml(folder / "model.yaml")["model"]["components"]}
    assert all("free" not in p for p in run_model["bulge"]["params"].values())  # the file's own settings
    assert run_model["bulge"].get("constraints")  # its |e| bound kept
    assert read_yaml(folder / "result.yaml")["fitted_only"] == ["disk"]
    assert list_history(tmp_path / "fits")[0]["status"].endswith("only disk")
    assert (folder / "imgblock.fits").exists()


@needs_data
def test_nuts_from_a_fit_in_the_history(tmp_path):
    """posterior.run_nuts: from a fit folder, a new folder (started_from it) with the posterior and corner.png."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.history import list_history
    from forwardgal.posterior import run_nuts
    from forwardgal.run import run
    from forwardgal.serialize import read_yaml

    cfg = example_config()
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 5}
    write_yaml(cfg, tmp_path / "model.yaml")
    fit_folder, *_ = run(tmp_path / "model.yaml", errors=False, plot=False, progress=False, only=["disk"])
    folder, post = run_nuts(fit_folder, warmup=15, samples=15, chains=1, progress=False)
    assert folder.parent == fit_folder.parent and folder.name.endswith("-nuts")
    info = read_yaml(folder / "result.yaml")
    assert info["started_from"] == fit_folder.name and info["fitted_only"] == ["disk"]
    assert info["loss"] == pytest.approx(read_yaml(fit_folder / "result.yaml")["loss"], rel=1e-9)  # same point
    par = info["nuts"]["parameters"]
    assert {n.split(".")[0] for n in par if not n.startswith("amp:")} == {"disk"}  # only the disk sampled
    p = par["disk.R_e"]
    assert p["p16"] <= p["median"] <= p["p84"]
    assert (folder / "corner.png").exists() and (folder / "imgblock.fits").exists()
    assert list_history(tmp_path / "fits")[-1]["status"].startswith("NUTS")
