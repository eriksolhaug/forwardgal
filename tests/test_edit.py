"""G2a: checked edits of a model spec (forwardgal.edit), and the constraint text."""

import pytest

import forwardgal  # noqa: F401
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.edit import SpecError, check, constraint_text, set_param, set_tie
from forwardgal.serialize import model_to_dict


@pytest.fixture
def spec():
    """Bulge + disk (source plane, e1/e2 with the |e| bound) + sky."""
    comps = [sersic_component("bulge", "source", -5.72, 0.19, 0.05, 4.0, 0.9, 30.0),
             sersic_component("disk", "source", -5.72, 0.18, 0.25, 1.0, 0.6, 30.0),
             sky_component()]
    return model_to_dict(ModelParams(comps))


def test_the_starting_spec_is_valid(spec):
    assert check(spec).size == 2 * 6 + 0  # sky has no shape parameters


def test_set_param_returns_a_new_checked_spec(spec):
    new = set_param(spec, "disk", "n", value=2.0, lo=0.5, hi=6.0)
    p = new["components"][1]["params"]["n"]
    assert p == {"value": 2.0, "lo": 0.5, "hi": 6.0}
    assert spec["components"][1]["params"]["n"]["value"] == 1.0  # input untouched


def test_lock_and_unlock(spec):
    locked = set_param(spec, "disk", "n", free=False)
    assert locked["components"][1]["params"]["n"]["free"] is False
    assert check(locked).size == check(spec).size - 1
    unlocked = set_param(locked, "disk", "n", free=True)
    assert "free" not in unlocked["components"][1]["params"]["n"]  # default left out of the file


def test_remove_a_bound_and_add_a_prior(spec):
    new = set_param(spec, "disk", "n", hi=None, prior_mu=1.0, prior_sigma=0.5)
    assert new["components"][1]["params"]["n"] == {"value": 1.0, "lo": 0.5, "prior_mu": 1.0, "prior_sigma": 0.5}


@pytest.mark.parametrize("fields, words", [
    (dict(value=9.0), "inside the bounds"),          # outside (0.5, 8): reported, not clipped
    (dict(lo=2.0), "inside the bounds"),             # new bound excludes the current value
    (dict(prior_mu=1.0), "prior_mu and prior_sigma"),
    (dict(value=None), "cannot be removed"),
])
def test_invalid_edits_raise_and_change_nothing(spec, fields, words):
    with pytest.raises(SpecError, match=words):
        set_param(spec, "disk", "n", **fields)
    check(spec)  # still the valid original


def test_e1_e2_cannot_be_locked_alone(spec):
    """The |e| < 0.905 bound needs e1, e2 free: the GUI greys these cells out."""
    with pytest.raises(SpecError, match="free, unbounded and untied"):
        set_param(spec, "disk", "e1", free=False)


def test_ties(spec):
    tied = set_tie(spec, "disk.x0", "bulge.x0", 0.01)
    assert tied["ties"] == {"disk.x0": ["bulge.x0", 0.01]}
    values = check(tied).unpack(check(tied).start())
    assert float(values["disk"]["x0"]) == pytest.approx(-5.72 + 0.01)
    assert set_tie(tied, "disk.x0", None)["ties"] == {}
    with pytest.raises(SpecError, match="itself tied"):
        set_tie(tied, "bulge.y0", "disk.x0")  # a chain
    with pytest.raises(SpecError, match="unknown parameter"):
        set_tie(spec, "disk.x0", "bulge.nope")


def test_unknown_names(spec):
    with pytest.raises(KeyError):
        set_param(spec, "clump", "n", value=1.0)
    with pytest.raises(TypeError):
        set_param(spec, "disk", "n", sigma=1.0)


# --- TODO(Erik): constraint_text --------------------------------------------------


@pytest.mark.parametrize("p, text", [
    ({"value": 0.3}, ""),
    ({"value": 4.0, "lo": 0.5, "hi": 8.0}, "U(0.5, 8)"),
    ({"value": 0.3, "lo": 0.01}, "> 0.01"),
    ({"value": -1.0, "hi": 0.0}, "< 0"),
    ({"value": 0.1, "prior_mu": 0.0, "prior_sigma": 0.4}, "N(0, 0.4)"),
    ({"value": 4.0, "lo": 0.5, "hi": 8.0, "prior_mu": 4.0, "prior_sigma": 1.0}, "U(0.5, 8), N(4, 1)"),
    ({"value": 2.0, "free": False, "lo": 0.5}, "> 0.5"),
])
def test_constraint_text(p, text):
    assert constraint_text(p) == text


# --- TODO(Erik): tie_candidates ----------------------------------------------------


@pytest.fixture
def three():
    """bulge, disk, clump (source plane), QSO (image plane), sky."""
    from forwardgal.components import point_component

    comps = [sersic_component("bulge", "source", -5.72, 0.19, 0.05, 4.0, 0.9, 30.0),
             sersic_component("disk", "source", -5.72, 0.18, 0.25, 1.0, 0.6, 30.0),
             sersic_component("clump", "source", -5.70, 0.25, 0.05, 1.0, 0.8, 0.0),
             point_component("QSO", "image", 17.75, 3.63), sky_component()]
    return model_to_dict(ModelParams(comps))


def test_tie_candidates_same_key_other_components_same_plane(three):
    from forwardgal.edit import tie_candidates

    assert tie_candidates(three, "disk", "x0") == ["bulge.x0", "clump.x0"]
    assert tie_candidates(three, "disk", "n") == ["bulge.n", "clump.n"]
    assert tie_candidates(three, "QSO", "x0") == []  # the only image-plane x0


def test_tie_candidates_unknown_name_is_a_key_error(three):
    from forwardgal.edit import tie_candidates

    with pytest.raises(KeyError, match="clmp"):
        tie_candidates(three, "clmp", "x0")


def test_tie_candidates_exclude_tied_sources(three):
    from forwardgal.edit import tie_candidates

    tied = set_tie(three, "bulge.x0", "clump.x0", 0.0)
    assert tie_candidates(tied, "disk", "x0") == ["clump.x0"]  # bulge.x0 is tied: no chains
    assert tie_candidates(tied, "bulge", "y0") == ["disk.y0", "clump.y0"]  # other keys unaffected


# --- adding and removing components (A1, A2) --------------------------------------


def test_new_name_is_the_first_unused(spec):
    from forwardgal.edit import add_sersic, new_name

    assert new_name(spec) == "sersic1"
    s = add_sersic(spec, "sersic1", "source", -5.7, 0.2, 0.05)
    assert new_name(s) == "sersic2" and new_name(s, "clump") == "clump1"


def test_add_sersic_appends_a_checked_component_without_flux(spec):
    from forwardgal.edit import add_sersic

    new = add_sersic(spec, "clump", "source", -5.70, 0.25, 0.05, q=0.8, pa_deg=0.0)
    c = new["components"][-1]
    assert c["name"] == "clump" and c["kind"] == "sersic" and c["plane"] == "source"
    assert c["params"]["R_e"] == {"value": 0.05, "lo": 0.01, "hi": 2.0}  # the source-plane defaults
    assert c["params"]["n"]["value"] == 1.0 and not {"flux", "mag"} & set(c["params"])
    assert len(spec["components"]) == 3  # input untouched
    assert check(new).size == check(spec).size + 6


@pytest.mark.parametrize("args, words", [
    (("disk", "source", 0.0, 0.0, 0.1), "already"),
    (("clump", "lens", 0.0, 0.0, 0.1), "plane"),
    (("clump", "source", 0.0, 0.0, 3.0), "inside the bounds"),  # R_e > 2" in the source plane
])
def test_add_sersic_refuses(spec, args, words):
    from forwardgal.edit import add_sersic

    with pytest.raises(SpecError, match=words):
        add_sersic(spec, *args)


def test_remove_component_drops_its_ties(three):
    from forwardgal.edit import remove_component

    tied = set_tie(set_tie(three, "clump.x0", "disk.x0", 0.0), "bulge.n", "disk.n", 0.0)
    tied = set_tie(tied, "bulge.y0", "clump.y0", 0.0)
    new = remove_component(tied, "clump")
    assert [c["name"] for c in new["components"]] == ["bulge", "disk", "QSO", "sky"]
    assert new["ties"] == {"bulge.n": ["disk.n", 0.0]}  # both clump ties gone, as target and as source
    assert len(tied["components"]) == 5  # input untouched


def test_remove_component_name_must_match_exactly(three):
    """Removing "disk" must not touch a component called e.g. "disk2" (compare names, not prefixes)."""
    from forwardgal.edit import add_sersic, remove_component

    s = add_sersic(three, "disk2", "source", -5.7, 0.2, 0.1)
    s = set_tie(s, "disk2.n", "bulge.n", 0.0)
    new = remove_component(s, "disk")
    assert "disk2" in [c["name"] for c in new["components"]] and "disk2.n" in new["ties"]
    with pytest.raises(KeyError):
        remove_component(three, "nope")


# --- (e1, e2) -> (q, PA) (G2e) -----------------------------------------------------


def test_to_qpa_keeps_the_shape_and_the_parameter_order(spec):
    from forwardgal.components import sersic_qpa
    from forwardgal.edit import to_qpa

    new = to_qpa(spec, "disk")
    p = new["components"][1]["params"]
    assert list(p) == ["x0", "y0", "R_e", "n", "q", "pa_deg"]  # q, PA where e1, e2 were
    assert p["q"]["value"] == pytest.approx(0.6) and p["pa_deg"]["value"] == pytest.approx(30.0)
    assert (p["q"]["lo"], p["q"]["hi"], p["pa_deg"]["hi"]) == (0.05, 1.0, 180.0)
    assert not new["components"][1].get("constraints")  # the |e| bound is gone (q's bounds replace it)
    old = check(spec).unpack(check(spec).start())["disk"]
    q, pa = (float(x) for x in sersic_qpa(old))
    v = check(new).unpack(check(new).start())["disk"]
    assert float(v["q"]) == pytest.approx(q, abs=1e-9) and float(v["pa_deg"]) == pytest.approx(pa % 180, abs=1e-6)
    assert check(new).size == check(spec).size  # still two shape parameters
    assert "e1" in spec["components"][1]["params"]  # input untouched
    assert to_qpa(new, "disk") == new  # already (q, PA): unchanged


@pytest.mark.parametrize("q, pa, expect_pa", [(1.0, 0.0, 0.0), (0.05, 120.0, 120.0), (0.5, 170.0, 170.0)])
def test_to_qpa_edges(q, pa, expect_pa):
    """Round (q = 1), at the q bound, and a PA that qpa_from_e returns negative (170 -> -10 -> 170)."""
    from forwardgal.components import e_from_qpa
    from forwardgal.edit import to_qpa

    s = model_to_dict(ModelParams([sersic_component("c", "image", 0.0, 0.0, 0.3, 1.0, 0.5, 0.0)]))
    e1, e2 = e_from_qpa(q, pa)
    s["components"][0]["params"]["e1"]["value"], s["components"][0]["params"]["e2"]["value"] = float(e1), float(e2)
    if q == 0.05:  # exactly at |e| = 0.905 is outside the (open) |e| bound: start a hair inside
        s["components"][0]["params"]["e1"]["value"] *= 1 - 1e-9
        s["components"][0]["params"]["e2"]["value"] *= 1 - 1e-9
    p = to_qpa(s, "c")["components"][0]["params"]
    assert 0.05 < p["q"]["value"] < 1.0 and p["q"]["value"] == pytest.approx(q, abs=2e-6)
    assert p["pa_deg"]["value"] == pytest.approx(expect_pa, abs=1e-6)


def test_to_qpa_unknown_name(spec):
    from forwardgal.edit import to_qpa

    with pytest.raises(KeyError):
        to_qpa(spec, "nope")


# --- Fit selected (G4) --------------------------------------------------------------


def test_hold_others_locks_everything_else(three):
    from forwardgal.edit import hold_others

    tied = set_tie(three, "clump.x0", "bulge.x0", 0.0)
    held = hold_others(tied, ["disk", "sky"])
    by = {c["name"]: c for c in held["components"]}
    assert all(p.get("free") is False for p in by["bulge"]["params"].values())
    assert "free" not in by["clump"]["params"]["x0"]  # tied: follows bulge.x0 (held anyway)
    assert by["clump"]["params"]["n"]["free"] is False
    assert "constraints" not in by["bulge"] and by["disk"]["constraints"]  # the held bulge's |e| bound dropped
    assert all("free" not in p for p in by["disk"]["params"].values())
    p = check(held)
    assert set(k.split(".")[0] for k in p.free_keys) == {"disk"}  # only the disk is fitted (sky: no shape)
    assert by["bulge"]["params"]["x0"]["value"] == three["components"][0]["params"]["x0"]["value"]
    with pytest.raises(KeyError):
        hold_others(three, ["nope"])


# --- modes (G6a) -------------------------------------------------------------------


def test_add_and_remove_fourier_modes(spec):
    from forwardgal.edit import add_mode, remove_mode

    s = add_mode(spec, "disk", "fourier", 1)
    disk = s["components"][1]
    assert disk["params"]["c1"] == {"value": 0.0} and disk["params"]["s1"] == {"value": 0.0}
    fb = next(k for k in disk["constraints"] if k["type"] == "fourier_bound")
    assert fb["ms"] == [1] and fb["limit"] == 0.95
    s = add_mode(s, "disk", "fourier", 4)
    assert next(k for k in s["components"][1]["constraints"] if k["type"] == "fourier_bound")["ms"] == [1, 4]
    assert check(s).size == check(spec).size + 4
    with pytest.raises(SpecError, match="already"):
        add_mode(s, "disk", "fourier", 1)
    with pytest.raises(SpecError, match="not allowed"):
        add_mode(s, "disk", "fourier", 2)  # m = 2 is q, PA
    s = remove_mode(remove_mode(s, "disk", "fourier", 1), "disk", "fourier", 4)
    assert not any(k["type"] == "fourier_bound" for k in s["components"][1]["constraints"])
    assert check(s).size == check(spec).size
    with pytest.raises(SpecError, match="no fourier mode"):
        remove_mode(s, "disk", "fourier", 3)


def test_bending_mode_fixes_its_frame_at_the_current_pa(spec):
    from forwardgal.edit import add_mode, remove_mode

    s = add_mode(spec, "disk", "bending", 3)
    disk = s["components"][1]
    assert disk["params"]["b3"] == {"value": 0.0} and disk["bend_pa_ref"] == pytest.approx(30.0)
    s2 = add_mode(s, "disk", "bending", 2)
    assert s2["components"][1]["bend_pa_ref"] == pytest.approx(30.0)
    s3 = remove_mode(remove_mode(s2, "disk", "bending", 3), "disk", "bending", 2)
    assert "bend_pa_ref" not in s3["components"][1]
    with pytest.raises(SpecError, match="Sersic"):
        add_mode(spec, "sky", "bending", 2)


def test_modes_go_before_the_flux(spec):
    from forwardgal.edit import add_mode

    spec["components"][1]["params"]["mag"] = {"value": 24.0}
    s = add_mode(spec, "disk", "fourier", 3)
    assert list(s["components"][1]["params"])[-3:] == ["c3", "s3", "mag"]


def test_set_mode(spec):
    """TODO(Erik): set_mode -- a_m, phi_m (degrees) -> c_m = a cos(m phi), s_m = a sin(m phi)."""
    import numpy as np

    from forwardgal.edit import add_mode, set_mode

    s = add_mode(spec, "disk", "fourier", 3)
    s = set_mode(s, "disk", 3, a=0.1, phi_deg=20.0)
    p = s["components"][1]["params"]
    assert p["c3"]["value"] == pytest.approx(0.1 * np.cos(np.radians(60.0)))
    assert p["s3"]["value"] == pytest.approx(0.1 * np.sin(np.radians(60.0)))
    s2 = set_mode(s, "disk", 3, a=0.2)  # the phase stays
    assert np.degrees(np.arctan2(s2["components"][1]["params"]["s3"]["value"],
                                 s2["components"][1]["params"]["c3"]["value"])) / 3 == pytest.approx(20.0)
    assert np.hypot(s2["components"][1]["params"]["c3"]["value"],
                    s2["components"][1]["params"]["s3"]["value"]) == pytest.approx(0.2)
    s3 = set_mode(s, "disk", 3, phi_deg=-10.0)  # the amplitude stays
    assert s3["components"][1]["params"]["c3"]["value"] == pytest.approx(0.1 * np.cos(np.radians(-30.0)))
    assert spec["components"][1]["params"].get("c3") is None  # input untouched
    with pytest.raises(SpecError):
        set_mode(s, "disk", 3, a=0.99)  # beyond the joint bound sum a_m < 0.95
    with pytest.raises(KeyError):
        set_mode(s, "disk", 1, a=0.1)  # no m = 1 mode
