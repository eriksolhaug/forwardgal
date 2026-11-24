import jax
import jax.numpy as jnp
import numpy as np
import pytest

import forwardgal  # noqa: F401
from forwardgal.components import (
    Component,
    ModelParams,
    pa_window,
    point_component,
    sersic_component,
    sky_component,
)
from forwardgal.params import Param


def model(ties=None):
    # (q, PA) parametrization here: these tests are about per-parameter bounds;
    # the (e1, e2) default has its own tests below.
    bulge = sersic_component("C1 bulge", "source", -5.73, 0.19, 0.08, 3.8, 0.58, -25.0, ellipticity="qpa",
                             overrides={"n": Param(3.8, lo=0.5, hi=6.0, prior_mu=4.0, prior_sigma=0.5)})
    disk = sersic_component("C2 disk", "source", -5.72, 0.20, 0.30, 1.0, 0.5, 40.0, ellipticity="qpa",
                            overrides={"n": Param(1.0, free=False)})
    cluster = sersic_component("cluster", "image", 15.93, 0.99, 0.5, 4.0, 0.8, 0.0, ellipticity="qpa")
    qso = point_component("QSO", "source", -5.732, 0.195,
                          overrides={"x0": Param(-5.732, free=False), "y0": Param(0.195, free=False)})
    return ModelParams([bulge, disk, cluster, qso, sky_component()], ties=ties)


def test_vector_layout_and_roundtrip():
    m = model()
    # bulge 6 + disk 5 (n fixed) + cluster 6 + qso 0 + sky 0 (default bounds)
    assert m.size == 17
    assert m.free_keys[:3] == ["C1 bulge.x0", "C1 bulge.y0", "C1 bulge.R_e"]
    v = m.unpack(m.start())
    assert float(v["C1 bulge"]["n"]) == pytest.approx(3.8)
    assert float(v["C2 disk"]["n"]) == 1.0  # fixed
    assert float(v["QSO"]["x0"]) == -5.732
    assert v["sky"] == {}


def test_ties():
    m = model(ties={"C1 bulge.x0": ("C2 disk.x0", 0.0), "C1 bulge.y0": ("C2 disk.y0", 0.01)})
    assert m.size == 15 and "C1 bulge.x0" not in m.free_keys
    u = m.start().at[m.free_keys.index("C2 disk.x0")].set(-5.5)
    v = m.unpack(u)
    assert float(v["C1 bulge"]["x0"]) == pytest.approx(-5.5)
    assert float(v["C1 bulge"]["y0"]) == pytest.approx(float(v["C2 disk"]["y0"]) + 0.01)
    assert "tied to C2 disk.x0" in m.describe()


def test_bad_ties_rejected():
    with pytest.raises(ValueError, match="mixes"):
        model(ties={"C1 bulge.x0": ("cluster.x0", 0.0)})
    with pytest.raises(ValueError, match="unknown"):
        model(ties={"C1 bulge.x0": ("C9.x0", 0.0)})
    with pytest.raises(ValueError, match="chains"):
        model(ties={"C1 bulge.x0": ("C2 disk.x0", 0.0), "C2 disk.x0": ("C1 bulge.y0", 0.0)})


def test_component_validation():
    with pytest.raises(ValueError, match="plane"):
        sersic_component("a", "lens", 0, 0, 1, 1, 0.5, 0)
    with pytest.raises(ValueError, match="needs parameters"):
        Component("a", "point", "source", {"x0": Param(0.0)})
    with pytest.raises(ValueError, match="sky"):
        Component("s", "sky", "source", {})
    with pytest.raises(ValueError, match="unique"):
        ModelParams([sky_component(), sky_component()])


def test_log_prior_and_jacobian():
    m = model()
    u = m.start()
    lp_opt = float(m.log_prior(u, jacobian=False))
    # only the bulge n has a prior: N(4.0, 0.5) at 3.8 -> -0.5 * 0.4^2 = -0.08
    assert lp_opt == pytest.approx(-0.08)
    jac = sum(float(m._lookup(k).log_jacobian(u[i])) for i, k in enumerate(m.free_keys))
    assert float(m.log_prior(u, jacobian=True)) == pytest.approx(lp_opt + jac)


def test_differentiable_and_jittable():
    m = model(ties={"C1 bulge.x0": ("C2 disk.x0", 0.0)})
    f = jax.jit(lambda u: m.unpack(u)["C1 bulge"]["R_e"] * m.unpack(u)["C1 bulge"]["x0"] + m.log_prior(u))
    g = jax.grad(f)(m.start())
    assert g.shape == (m.size,) and np.all(np.isfinite(np.asarray(g)))


def test_default_constraints():
    c = sersic_component("a", "source", 0, 0, 1, 1, 0.5, -25.0, ellipticity="qpa")
    assert c.params["R_e"].lo == 0.01
    assert (c.params["n"].lo, c.params["n"].hi) == (0.5, 8.0)
    assert c.params["pa_deg"].hi == 180.0 and c.params["pa_deg"].lo == pytest.approx(0.0, abs=1e-5)
    assert c.params["pa_deg"].value == pytest.approx(155.0)  # -25 wrapped into [0, 180)
    assert c.params["x0"].lo is None and c.params["x0"].hi is None
    assert sersic_component("b", "source", 0, 0, 1, 1, 0.5, 0.0, ellipticity="qpa").params["pa_deg"].value == 0.0


def test_pa_window_only_when_asked():
    plain = sersic_component("a", "source", 0, 0, 1, 1, 0.5, 30.0, ellipticity="qpa")
    assert plain.params["pa_deg"].hi == 180.0  # the default range
    windowed = sersic_component("b", "source", 0, 0, 1, 1, 0.5, 30.0, ellipticity="qpa",
                                overrides={"pa_deg": pa_window(30.0)})
    assert (windowed.params["pa_deg"].lo, windowed.params["pa_deg"].hi) == (-60.0, 120.0)


# --- "centre within N" constraints -------------------------------------------------

from forwardgal.components import lens_jacobian, within  # noqa: E402
from test_traced import linear_field  # noqa: E402


def constrained(mode, unit="arcsec", start=(-5.70, 0.20), **kw):
    c = sersic_component("host", "source", *start, 0.1, 1.0, 0.6, 0.0, ellipticity="qpa")
    pc = within(c, (-5.72, 0.19), 0.05 if unit == "arcsec" else 2.0, unit, mode=mode, **kw)
    return ModelParams([c, sky_component()], constraints=[pc]), pc


def test_bounds_mode_never_leaves_the_region():
    m, pc = constrained("bounds")
    i, j = m.free_keys.index("host.x0"), m.free_keys.index("host.y0")
    rng = np.random.default_rng(8)
    for _ in range(200):
        u = m.start().at[i].set(rng.normal(0, 30)).at[j].set(rng.normal(0, 30))
        v = m.unpack(u)
        assert np.hypot(float(v["host"]["x0"]) + 5.72, float(v["host"]["y0"]) - 0.19) < 0.05


def test_bounds_mode_start_roundtrip_and_jacobian():
    m, pc = constrained("bounds")
    v = m.unpack(m.start())
    assert float(v["host"]["x0"]) == pytest.approx(-5.70, abs=1e-12)
    assert float(v["host"]["y0"]) == pytest.approx(0.20, abs=1e-12)
    i, j = m.free_keys.index("host.x0"), m.free_keys.index("host.y0")
    u0 = m.start().at[i].set(0.7).at[j].set(-1.3)
    xy = lambda a: jnp.stack([m.unpack(u0.at[i].set(a[0]).at[j].set(a[1]))["host"][k] for k in ("x0", "y0")])  # noqa: E731
    J = np.asarray(jax.jacfwd(xy)(jnp.array([0.7, -1.3])))
    others = sum(float(m._lookup(k).log_jacobian(u0[n])) for n, k in enumerate(m.free_keys) if k not in ("host.x0", "host.y0"))
    lp = float(m.log_prior(u0, jacobian=True)) - float(m.log_prior(u0, jacobian=False))
    assert lp - others == pytest.approx(np.log(abs(np.linalg.det(J))), abs=1e-10)


def test_gaussian_mode_prior():
    m, pc = constrained("gaussian")
    v = m.unpack(m.start())
    # offset (0.02, 0.01) / 0.05 -> |w|^2 = 0.2 -> log prior -0.1
    assert float(m.log_prior(m.start(), jacobian=False)) == pytest.approx(-0.1)
    assert "Gaussian" in m.describe() and 'within 0.05"' in m.describe()


def test_px_constraint_for_source_component_is_a_circle_in_the_image_plane():
    # Linear lens beta = (0.1 theta_x, 1.0 theta_y): a 10x stretch along x.
    field = linear_field(0.45, 0.45)
    A = lens_jacobian(field, (3.0, -1.0))
    np.testing.assert_allclose(A, np.diag([0.1, 1.0]), atol=1e-9)
    # Start inside: 0.0006" in beta_x = 0.006" = 0.2 px in the image plane.
    # (A start 0.02" away in beta_x would be 6.7 px away in theta: refused.)
    m, pc = constrained("bounds", unit="px", start=(-5.7194, 0.19), pixscale=0.03, lens_matrix=A)
    i, j = m.free_keys.index("host.x0"), m.free_keys.index("host.y0")
    for ang in np.linspace(0, 2 * np.pi, 13):
        u = m.start().at[i].set(1e4 * np.cos(ang)).at[j].set(1e4 * np.sin(ang))  # on the boundary
        v = m.unpack(u)
        dbx, dby = float(v["host"]["x0"]) + 5.72, float(v["host"]["y0"]) - 0.19
        dtx, dty = dbx / 0.1, dby / 1.0  # back to the image plane
        assert np.hypot(dtx, dty) / 0.03 == pytest.approx(2.0, rel=1e-6)  # exactly 2 data pixels
    assert "2 px (image plane)" in m.describe()


def test_constraint_validation():
    c = sersic_component("host", "source", -5.70, 0.20, 0.1, 1.0, 0.6, 0.0,
                         overrides={"x0": Param(-5.70, lo=-6.0, hi=-5.0)})
    with pytest.raises(ValueError, match="unbounded"):
        ModelParams([c], constraints=[within(c, (-5.72, 0.19), 0.05, "arcsec")])
    c2 = sersic_component("host", "source", -5.50, 0.20, 0.1, 1.0, 0.6, 0.0)
    with pytest.raises(ValueError, match="outside"):
        ModelParams([c2], constraints=[within(c2, (-5.72, 0.19), 0.05, "arcsec")]).start()
    with pytest.raises(ValueError, match="lens_matrix"):
        within(c2, (-5.72, 0.19), 2.0, "px", pixscale=0.03)


def test_unbounded_overrides_allowed_with_warning():
    # Any parameter may be unbounded; removing validity bounds warns.
    with pytest.warns(UserWarning, match="R_e may become"):
        c = sersic_component("a", "source", 0, 0, 0.1, 1, 0.5, 0, overrides={"R_e": Param(0.1)})
    assert c.params["R_e"].lo is None
    with pytest.warns(UserWarning, match="0.5-8"):
        sersic_component("b", "source", 0, 0, 0.1, 1, 0.5, 0, overrides={"n": Param(1.0)})
    # unbounded positions and PA are the default and do not warn
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        sersic_component("c", "image", 0, 0, 0.1, 1, 0.5, 0)


def test_prior_residuals_match_log_prior():
    # sum of squared prior residuals = -2 log_prior (no Jacobian), incl. a
    # Gaussian "within" constraint.
    c = sersic_component("host", "source", -5.70, 0.20, 0.1, 1.0, 0.6, 0.0,
                         overrides={"n": Param(1.0, lo=0.5, hi=6.0, prior_mu=1.5, prior_sigma=0.3),
                                    "R_e": Param(0.1, lo=0.0, prior_mu=0.12, prior_sigma=0.01)})
    m = ModelParams([c, sky_component()], constraints=[within(c, (-5.72, 0.19), 0.05, "arcsec", mode="gaussian")])
    u = m.start() + 0.1
    r = m.prior_residuals(u)
    assert r.shape == (4,)  # n, R_e, w1, w2
    assert float(jnp.sum(r**2)) == pytest.approx(-2 * float(m.log_prior(u, jacobian=False)), rel=1e-12)
    assert ModelParams([sky_component()]).prior_residuals(jnp.zeros(0)).shape == (0,)



# --- (e1, e2) ellipticity -------------------------------------------------------------

from forwardgal.components import EPS_MAX, e_from_qpa, qpa_from_e, regularize, sersic_qpa  # noqa: E402


@pytest.mark.parametrize("q, pa", [(0.6, 30.0), (0.2, 170.0), (0.95, 0.0), (0.5, 90.0)])
def test_e_qpa_roundtrip(q, pa):
    e1, e2 = e_from_qpa(q, pa)
    q2, pa2 = qpa_from_e(jnp.asarray(e1), jnp.asarray(e2))
    assert float(q2) == pytest.approx(q, abs=1e-12)
    assert float(pa2) % 180 == pytest.approx(pa % 180, abs=1e-9)


def test_pa_and_pa_plus_180_are_the_same_point():
    np.testing.assert_allclose(e_from_qpa(0.5, 20.0), e_from_qpa(0.5, 200.0), atol=1e-12)
    np.testing.assert_allclose(e_from_qpa(0.5, 179.0), e_from_qpa(0.5, -1.0), atol=1e-12)


def test_round_shape_has_finite_gradients():
    g = jax.grad(lambda e: jnp.sum(jnp.stack(qpa_from_e(e[0], e[1]))))(jnp.zeros(2))
    assert np.all(np.isfinite(np.asarray(g)))
    q, _ = qpa_from_e(jnp.asarray(0.0), jnp.asarray(0.0))
    assert float(q) == 1.0


def test_e1e2_is_default_with_joint_bound():
    c = sersic_component("host", "source", -5.7, 0.2, 0.1, 1.0, 0.6, 30.0)
    assert set(c.params) == {"x0", "y0", "R_e", "n", "e1", "e2"} and c.names[-2:] == ("e1", "e2")
    m = ModelParams([c, sky_component()])
    v = m.unpack(m.start())
    q, pa = sersic_qpa(v["host"])
    assert float(q) == pytest.approx(0.6) and float(pa) % 180 == pytest.approx(30.0)
    i, j = m.free_keys.index("host.e1"), m.free_keys.index("host.e2")
    rng = np.random.default_rng(9)
    for _ in range(100):  # |e| never reaches the bound, i.e. q > 0.05
        u = m.start().at[i].set(rng.normal(0, 50)).at[j].set(rng.normal(0, 50))
        vv = m.unpack(u)["host"]
        assert np.hypot(float(vv["e1"]), float(vv["e2"])) < EPS_MAX
    assert "|e| < 0.905" in m.describe() and "start q = 0.600" in m.describe()


def test_e1e2_joint_jacobian_matches_autodiff():
    c = sersic_component("host", "source", -5.7, 0.2, 0.1, 1.0, 0.6, 30.0)
    m = ModelParams([c])
    i, j = m.free_keys.index("host.e1"), m.free_keys.index("host.e2")
    u0 = m.start().at[i].set(0.4).at[j].set(-0.9)
    ee = lambda a: jnp.stack([m.unpack(u0.at[i].set(a[0]).at[j].set(a[1]))["host"][k] for k in ("e1", "e2")])  # noqa: E731
    J = np.asarray(jax.jacfwd(ee)(jnp.array([0.4, -0.9])))
    others = sum(float(m._lookup(k).log_jacobian(u0[n])) for n, k in enumerate(m.free_keys) if k not in ("host.e1", "host.e2"))
    lp = float(m.log_prior(u0, jacobian=True)) - float(m.log_prior(u0, jacobian=False))
    assert lp - others == pytest.approx(np.log(abs(np.linalg.det(J))), abs=1e-10)


def test_regularize_adds_weak_shape_priors():
    c = regularize(sersic_component("host", "source", -5.7, 0.2, 0.1, 2.0, 0.6, 30.0))
    assert (c.params["n"].prior_mu, c.params["n"].prior_sigma) == (2.0, 2.0)
    assert c.params["R_e"].prior_sigma == pytest.approx(0.1)
    assert (c.params["e1"].prior_mu, c.params["e1"].prior_sigma) == (0.0, 0.4)
    m = ModelParams([c])  # still valid with the |e| bound
    assert np.isfinite(float(m.log_prior(m.start())))


# --- Fourier modes (M12d) ----------------------------------------------------

from forwardgal.components import FOURIER_SUM_MAX, FourierBound, aphi_from_cs, modes_cs  # noqa: E402


def test_sersic_with_modes_has_cs_pairs_and_a_bound():
    c = sersic_component("host", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes={1: (0.1, 30.0), 4: (0.05, 10.0)})
    assert c.names[-4:] == ("c1", "s1", "c4", "s4")
    assert c.modes == (1, 4)
    assert any(isinstance(fb, FourierBound) for fb in c.constraints)
    # starting values are GALFIT (a, phi) converted to (c, s)
    a, phi = aphi_from_cs(c.params["c4"].value, c.params["s4"].value, 4)
    assert float(a) == pytest.approx(0.05) and float(phi) == pytest.approx(10.0)


def test_mode_list_starts_at_zero_and_unpacks_to_zero():
    c = sersic_component("host", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes=[1, 3, 4])
    params = ModelParams([c])
    v = params.unpack(params.start())["host"]
    for k in ("c1", "s1", "c3", "s3", "c4", "s4"):
        assert float(v[k]) == 0.0


@pytest.mark.parametrize("bad", [{2: (0.1, 0.0)}, {5: (0.1, 0.0)}])
def test_only_modes_1_3_4_allowed(bad):
    with pytest.raises(ValueError, match="not allowed"):
        sersic_component("host", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes=bad)


def test_starting_amplitudes_must_respect_the_bound():
    with pytest.raises(ValueError, match="sum"):
        sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes={1: (0.6, 0.0), 3: (0.4, 0.0)})


def test_fourier_bound_holds_for_any_u():
    """Random and extreme unconstrained values: sum a_m stays below the limit,
    and a single mode can approach it."""
    fb = FourierBound("h", (1, 3, 4))
    rng = np.random.default_rng(0)
    for scale in (0.1, 1.0, 10.0, 1e3, 1e6):
        for _ in range(50):
            y = np.asarray(fb.to_physical(jnp.asarray(scale * rng.normal(size=6)))).reshape(3, 2)
            assert np.hypot(y[:, 0], y[:, 1]).sum() < FOURIER_SUM_MAX
    y = np.asarray(fb.to_physical(jnp.array([1e6, 0, 0, 0, 0, 0]))).reshape(3, 2)
    assert np.hypot(y[:, 0], y[:, 1]).sum() > 0.94


def test_fourier_bound_inverse_roundtrip():
    fb = FourierBound("h", (1, 3, 4))
    for y in ([0.1, -0.05, 0.02, 0.0, 0.0, 0.03], [0.7, 0.2, 0.0, 0.0, 0.0, 0.0], [0.0] * 6):
        w = fb.to_unconstrained(np.array(y))
        np.testing.assert_allclose(np.asarray(fb.to_physical(jnp.asarray(w))), y, atol=1e-12)


def test_fourier_bound_log_jacobian_matches_autodiff():
    fb = FourierBound("h", (1, 3, 4))
    for w in (jnp.zeros(6), jnp.array([0.3, -1.2, 0.05, 0.0, 2.0, 0.4])):
        _, logdet = jnp.linalg.slogdet(jax.jacfwd(fb.to_physical)(w))
        assert float(fb.log_jacobian(w)) == pytest.approx(float(logdet), rel=1e-10, abs=1e-12)


def test_fourier_bound_gradients_finite_at_zero():
    fb = FourierBound("h", (1, 4))
    g = jax.jacfwd(fb.to_physical)(jnp.zeros(4))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.all(np.isfinite(np.asarray(jax.grad(fb.log_jacobian)(jnp.zeros(4)))))


def test_aphi_style_uses_per_mode_bounds_and_no_joint_bound():
    c = sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes={1: (0.1, 30.0), 3: (0.0, 0.0)},
                         mode_style="aphi")
    assert c.names[-4:] == ("a1", "phi1", "a3", "phi3")
    assert c.params["a1"].hi == pytest.approx(FOURIER_SUM_MAX / 2)
    assert not any(isinstance(fb, FourierBound) for fb in c.constraints)
    # the same shape in both styles gives the same Cartesian modes
    cs = sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes={1: (0.1, 30.0), 3: (0.0, 0.0)})
    va = ModelParams([c]).unpack(ModelParams([c]).start())["h"]
    vc = ModelParams([cs]).unpack(ModelParams([cs]).start())["h"]
    for x, y in zip(modes_cs(va, (1, 3))[1:], modes_cs(vc, (1, 3))[1:]):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)


def test_fixing_a_mode_parameter_drops_the_joint_bound_with_warning():
    with pytest.warns(UserWarning, match="NOT applied"):
        c = sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes=[1],
                             overrides={"s1": Param(0.0, free=False)})
    assert not any(isinstance(fb, FourierBound) for fb in c.constraints)


def test_jittered_starts_respect_the_fourier_bound():
    from forwardgal.fit import _jittered_start

    c = sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, modes={1: (0.5, 0.0), 3: (0.3, 0.0)})
    params = ModelParams([c])
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = params.unpack(_jittered_start(params, rng))["h"]
        a = sum(float(jnp.hypot(v[f"c{m}"], v[f"s{m}"])) for m in (1, 3))
        assert a < FOURIER_SUM_MAX


# --- bending modes (M12f) -----------------------------------------------------

from forwardgal.components import BEND_M, bending_of, galfit_bend_sign  # noqa: E402


def test_sersic_with_bending_has_b_params_and_pa_ref():
    c = sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 120.0, modes=[1], bending={2: 0.02, 3: 0.0})
    assert c.names[-2:] == ("b2", "b3") and c.bends == (2, 3)
    assert c.bend_pa_ref == 120.0
    v = ModelParams([c]).unpack(ModelParams([c]).start())["h"]
    b = bending_of(v, c)
    assert b.m == (2, 3) and float(b.a[0]) == pytest.approx(0.02) and b.pa_ref == 120.0


@pytest.mark.parametrize("m", [1, 6])
def test_only_bending_2_to_5_allowed(m):
    assert m not in BEND_M
    with pytest.raises(ValueError, match="not allowed"):
        sersic_component("h", "source", 0.0, 0.0, 0.3, 1.0, 0.7, 20.0, bending=[m])


def test_galfit_bend_sign_flips_even_modes_across_galfits_fold():
    # start PA 120 (outside GALFIT's (-90, 90]): our frame is turned w.r.t. GALFIT's
    assert float(galfit_bend_sign(120.0, 120.0, 2)) == -1.0
    assert float(galfit_bend_sign(120.0, 120.0, 3)) == 1.0
    assert float(galfit_bend_sign(30.0, 30.0, 2)) == 1.0
