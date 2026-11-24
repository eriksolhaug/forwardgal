"""The Qt GUI (G1b), run without a screen (QT_QPA_PLATFORM=offscreen)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import forwardgal  # noqa: E402,F401
from test_config import needs_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_window_starts_empty(app):
    from forwardgal.gui.app import MainWindow

    w = MainWindow()
    assert [a.text() for a in w.menuBar().actions()] == ["&File", "&Settings", "&View", "S&cale", "S&mooth"]
    assert [a.text() for a in w.tools.actions()] == ["Select / move", "Pan / zoom", "Draw ellipse", "Edit mask"]
    assert w.setup is None and w.rois == {}
    w.render()  # nothing loaded: no error


@pytest.fixture(scope="module")
def loaded(app, tmp_path_factory):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.gui.app import MainWindow
    from forwardgal.serialize import write_yaml

    d = tmp_path_factory.mktemp("gui")
    write_yaml(example_config(), d / "model.yaml")
    w = MainWindow()
    assert w.load(d / "model.yaml")
    yield w, d
    # Close and delete the window here, not whenever Python's garbage collector gets to it: a Qt
    # window torn down by the collector in the middle of a later test can deadlock (seen 2026-11-22).
    w.close()
    w.deleteLater()
    app.processEvents()


@needs_data
def test_load_shows_both_planes_and_matches_the_fitter(loaded):
    import jax.numpy as jnp

    w, _ = loaded
    assert set(w.rois) == {"bulge", "disk"}  # one ellipse per source-plane Sersic
    assert w.image.image.image.shape == w.live.data.shape  # the render box
    assert w.source.image.image.shape == (200, 200)
    c2, dof = w.setup.model.chi2_dof(jnp.asarray(w.setup.model.params.start()))
    assert w.result.chi2 == pytest.approx(c2, rel=1e-8) and w.result.dof == dof
    assert "χ²/ν" in w.fitinfo.text()
    for k in range(3):  # Data, Model, Residual
        w.mode.button(k).click()
    assert w._shown["image"] == ("Residual", (-5.0, 5.0))  # default: linear, -5 .. 5 sigma
    w.mode.button(0).click()
    w.resize(1500, 860)
    (ROOT / "outputs").mkdir(exist_ok=True)
    w.grab().save(str(ROOT / "outputs" / "g1b_gui.png"))


def default_view(w, arr):
    """What the default scaling shows for an image-plane array (not Residual): asinh(v / noise) between the
    0.5 and 99.7 % points of the (asinh) data in the mask, as 0 .. 1."""
    t = np.arcsinh(w.live.data[w.live.mask] / w._soft)
    lo, hi = np.percentile(t, [0.5, 99.7])
    return np.clip((np.arcsinh(np.asarray(arr) / w._soft) - lo) / (hi - lo), 0, 1)


@needs_data
def test_ellipses_sit_on_the_components(loaded):
    """The centre of each ROI (as pyqtgraph draws it) is the component's (x0, y0)."""
    from PySide6.QtCore import QPointF

    w, _ = loaded
    for name, roi in w.rois.items():
        st = roi.getState()
        c = roi.mapToParent(QPointF(st["size"].x() / 2, st["size"].y() / 2))
        v = w.live.values[name]
        assert (c.x(), c.y()) == pytest.approx((v["x0"], v["y0"]), abs=1e-9)


@needs_data
def test_dragging_an_ellipse_edits_the_spec_and_rerenders(loaded):
    from forwardgal.gui.geometry import roi_to_ellipse

    try:
        roi_to_ellipse((0.0, 0.0), (2.0, 1.0), 0.0)
    except NotImplementedError:
        pytest.skip("TODO(Erik): roi_to_ellipse")
    w, _ = loaded
    roi = w.rois["disk"]
    x_before = w.live.values["disk"]["x0"]
    chi2_before = w.result.chi2
    roi.setPos(roi.pos() + type(roi.pos())(0.05, 0.0))  # 0.05" east
    assert w.timer.isActive()  # a render is scheduled, not done yet
    w.render()  # what the timer would do
    assert w.live.values["disk"]["x0"] == pytest.approx(x_before + 0.05, abs=1e-9)
    assert w.result.chi2 != chi2_before
    roi.setPos(roi.pos() - type(roi.pos())(0.05, 0.0))  # put it back
    w.render()


@needs_data
def test_save_and_reload_gives_the_same_config(loaded):
    from forwardgal.serialize import read_yaml

    w, d = loaded
    w.save(d / "again.yaml")
    saved = read_yaml(d / "again.yaml")
    assert saved == w.cfg
    qso = saved["model"]["components"][-1]  # the old quasar: section, saved as a point with its RA/Dec
    assert "quasar" not in saved and qso["kind"] == "point" and list(qso["params"]) == ["mag"]  # + its flux
    assert (qso["ra"], qso["dec"]) == (178.3352306, 7.9335254)


@needs_data
def test_coordinate_readout_matches_the_lens(loaded):
    """The arm point traced in M12f: theta (17.80, 2.60) -> beta (-5.836, -0.119)."""
    from forwardgal.gui.app import theta_text

    w, _ = loaded
    text = theta_text(w.setup, 17.80, 2.60)
    assert "theta = (+17.800, +2.600)" in text
    assert "beta = (-5.8359, -0.1186)" in text
    assert np.isfinite(w.result.chi2)


@needs_data
def test_components_panel_lists_and_shows_the_selected_component(loaded):
    """G2b: list (incl. the quasar as a point), parameter card of the selected component, live values."""
    w, _ = loaded
    p = w.panel
    assert p.names == ["bulge", "disk", "sky", "QSO 1A"]  # spec order; the quasar is a point component
    assert p.current == "bulge"
    value, constraint, lock = p._cells["n"]
    assert value.text() == f"{w.live.values['bulge']['n']:.3f}" and constraint.text() == "U(0.5, 8)"
    assert not value.isReadOnly() and lock.isEnabled()  # values editable (G2c), lock boxes (G2d)
    p.list.setCurrentRow(3)  # the user picks the quasar
    assert p.current == "QSO 1A" and list(p._cells) == ["x0", "y0", "mag"]  # flux parameter (F4a)
    assert p._cells["x0"][2].isChecked() and p._cells["x0"][0].isReadOnly()  # locked position
    assert p._cells["mag"][0].text() == f"{w.live.values['QSO 1A']['mag']:.3f}"
    p.list.setCurrentRow(1)  # the disk; drag its ellipse: the card follows after the render
    roi = w.rois["disk"]
    roi.setPos(roi.pos() + type(roi.pos())(0.05, 0.0))
    w.render()
    assert p._cells["x0"][0].text() == f"{w.live.values['disk']['x0']:+.4f}"
    roi.setPos(roi.pos() - type(roi.pos())(0.05, 0.0))
    w.render()
    w.resize(1500, 860)
    w.grab()  # a never-shown window is laid out by its first grab, which paints the old layout
    w.grab().save(str(ROOT / "outputs" / "g2b_gui.png"))


@needs_data
def test_selection_goes_both_ways(loaded):
    """G2c: a row highlights its ellipse; clicking/dragging an ellipse selects its row."""
    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(1)  # the user picks the disk in the list
    assert w.rois["disk"].pen.width() > w.rois["bulge"].pen.width()
    assert w.rois["disk"].zValue() > w.rois["bulge"].zValue()
    w.rois["bulge"].sigRegionChangeStarted.emit(w.rois["bulge"])  # the user starts dragging the bulge
    assert p.current == "bulge" and p.list.currentRow() == 0
    assert w.rois["bulge"].pen.width() > w.rois["disk"].pen.width()


@needs_data
def test_typing_a_value_edits_the_spec_moves_the_ellipse_and_refuses_bad_input(loaded):
    """G2c: a value typed in the card is a checked spec edit; the ellipse follows."""
    from forwardgal.gui.rows import parse_value

    try:
        parse_value("1")
    except NotImplementedError:
        pytest.skip("TODO(Erik): parse_value")
    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(1)  # the disk
    spec0 = w.spec
    field = p._cells["R_e"][0]
    field.setText("0.3")
    field.editingFinished.emit()  # as on Enter
    assert w.live.values["disk"]["R_e"] == pytest.approx(0.3)
    st = w.rois["disk"].getState()
    assert max(st["size"].x(), st["size"].y()) == pytest.approx(0.6)  # 2 R_e
    q_field = p._cells["q"][0]
    q_field.setText("0.7")  # (e1, e2) component: through set_ellipse
    q_field.editingFinished.emit()
    st = w.rois["disk"].getState()
    assert min(st["size"].x(), st["size"].y()) / max(st["size"].x(), st["size"].y()) == pytest.approx(0.7)
    before = w.spec
    for bad in ("9", "abc"):  # n outside U(0.5, 8); not a number
        n_field = p._cells["n"][0]
        n_field.setText(bad)
        n_field.editingFinished.emit()
        assert w.spec is before and n_field.text() == f"{w.live.values['disk']['n']:.3f}"  # unchanged, reverted
    assert "disk.n" in w.statusBar().currentMessage()
    w.spec = spec0  # put the model back for other tests
    w.render()


@needs_data
def test_lock_boxes_fix_and_free_parameters(loaded):
    """G2d: a lock box sets the parameter's free flag; nu follows; not-lockable boxes are disabled."""
    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(1)  # the disk
    dof0, chi0 = w.result.dof, w.result.chi2
    lock = p._cells["n"][2]
    assert lock.isEnabled() and not lock.isChecked()
    lock.click()  # the user locks n
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    assert disk["params"]["n"]["free"] is False and lock.isChecked()
    assert w.result.dof == dof0 + 1 and w.result.chi2 == pytest.approx(chi0, rel=1e-12)  # same model
    assert "locked at" in w.statusBar().currentMessage()
    lock.click()  # and frees it again
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    assert "free" not in disk["params"]["n"] and w.result.dof == dof0
    assert p._cells["q"][2].isEnabled()  # (e1, e2): lockable since G2e (converts to (q, PA) first)
    p.list.setCurrentRow(3)  # the quasar: position locked, and it stays so
    assert p._cells["x0"][2].isChecked() and not p._cells["x0"][2].isEnabled()
    assert p._cells["mag"][2].isEnabled()  # a flux is a parameter: lockable (F4a)


@needs_data
def test_constraint_dialog_sets_bounds_and_priors_or_says_why_not(loaded):
    """G2d-2: click a constraint, fill the dialog; refused edits keep the dialog open, model unchanged."""
    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(1)  # the disk
    spec0 = w.spec
    p._cells["n"][1].click()  # the user clicks "U(0.5, 8)"
    dlg = w.dialog
    assert dlg.isVisible() and dlg.fields["lo"].text() == "0.5" and dlg.fields["hi"].text() == "8"
    assert dlg.fields["prior_mu"].text() == ""  # no prior yet
    dlg.fields["lo"].setText("2")  # excludes the current n = 1: refused
    dlg.ok()
    assert dlg.isVisible() and "inside the bounds" in dlg.error.text() and w.spec is spec0
    dlg.fields["lo"].setText("0.5")
    dlg.fields["hi"].setText("6")
    dlg.fields["prior_mu"].setText("1")  # mu without sigma: refused
    dlg.ok()
    assert dlg.isVisible() and "prior_sigma" in dlg.error.text() and w.spec is spec0
    dlg.fields["prior_sigma"].setText("0.5")
    dlg.ok()
    assert not dlg.isVisible()
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    assert disk["params"]["n"] == {"value": 1.0, "lo": 0.5, "hi": 6.0, "prior_mu": 1.0, "prior_sigma": 0.5}
    assert p._cells["n"][1].text() == "U(0.5, 6), N(1, 0.5)"
    p._cells["n"][1].click()
    w.dialog.fields["lo"].clear()  # remove the bounds and the prior
    w.dialog.fields["hi"].clear()
    w.dialog.fields["prior_mu"].clear()
    w.dialog.fields["prior_sigma"].clear()
    w.dialog.ok()
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    assert disk["params"]["n"] == {"value": 1.0} and p._cells["n"][1].text() == "none"
    assert p._cells["q"][1].isEnabled()  # (e1, e2): since G2e a click converts to (q, PA), then opens
    w.spec = spec0
    w.render()


@needs_data
def test_tie_dialog_ties_without_moving_and_untie_keeps_the_value(loaded):
    """G2d-3: + Tie… -> dialog (offset = current difference) -> tied; untie keeps the value."""
    from forwardgal.edit import tie_candidates

    try:
        tie_candidates(loaded[0].spec, "disk", "x0")
    except NotImplementedError:
        pytest.skip("TODO(Erik): tie_candidates")
    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(1)  # the disk
    spec0, chi0, dof0 = w.spec, w.result.chi2, w.result.dof
    x_disk, x_bulge = w.live.values["disk"]["x0"], w.live.values["bulge"]["x0"]
    p.tie_button.click()
    dlg = w.dialog
    assert dlg.target.currentData() == "x0" and dlg.source.currentText() == "bulge.x0"
    assert float(dlg.offset.text()) == pytest.approx(x_disk - x_bulge, abs=1e-6)  # nothing moves
    dlg.ok()
    assert not dlg.isVisible() and w.spec["ties"] == {"disk.x0": ["bulge.x0", pytest.approx(x_disk - x_bulge, abs=1e-6)]}
    assert w.live.values["disk"]["x0"] == pytest.approx(x_disk, abs=1e-6)
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9)  # the current offset: the model did not move
    assert w.result.dof == dof0 + 1  # disk.x0 is no longer a fit parameter
    assert p._cells["x0"][0].isReadOnly()  # tied: follows bulge.x0, not typed
    # Dragging the bulge now moves the disk's ellipse too.
    roi = w.rois["bulge"]
    roi.setPos(roi.pos() + type(roi.pos())(0.05, 0.0))
    w.render()
    assert w.live.values["disk"]["x0"] == pytest.approx(x_disk + 0.05, abs=1e-6)
    roi.setPos(roi.pos() - type(roi.pos())(0.05, 0.0))
    w.render()
    # Untie: the disk keeps its (tied) value and is free again.
    w._untie("disk.x0")
    assert w.spec["ties"] == {} and w.live.values["disk"]["x0"] == pytest.approx(x_disk, abs=1e-6)
    # A tie to itself is not offered; one that is refused stays open with the reason.
    p.tie_button.click()
    assert "disk.x0" not in [w.dialog.source.itemText(i) for i in range(w.dialog.source.count())]
    w.dialog.offset.setText("abc")
    w.dialog.ok()
    assert w.dialog.isVisible() and "offset" in w.dialog.error.text()
    w.dialog.reject()
    w.spec = spec0
    w.render()


@needs_data
def test_fluxes_are_parameters_in_the_gui_and_fit_fluxes_sets_them(loaded):
    """F4a: loading adds flux parameters (one linear solve); dragging never changes a flux; the button does."""
    from forwardgal.photometry import mag_to_flux

    w, _ = loaded
    p = w.panel
    for c in w.spec["components"]:  # every component has its flux parameter now
        assert ("flux" if c["kind"] == "sky" else "mag") in c["params"]
    mags = {n: v.get("mag") for n, v in w.live.values.items()}
    roi = w.rois["disk"]
    roi.setPos(roi.pos() + type(roi.pos())(0.05, 0.0))  # drag the disk
    w.render()
    assert {n: v.get("mag") for n, v in w.live.values.items()} == mags  # fluxes untouched
    assert w.result.amplitudes["disk"] == pytest.approx(mag_to_flux(mags["disk"], w.live.zeropoint))
    p.list.setCurrentRow(3)  # lock the quasar's flux, then "Fit fluxes" for all: it must stay
    p._cells["mag"][2].click()
    p.list.selectAll()  # (Ctrl/⌘+A)
    assert p.fit_fluxes.text() == f"Fit fluxes ({len(p.names)})"
    p.fit_fluxes.click()
    assert w.live.values["QSO 1A"]["mag"] == mags["QSO 1A"]
    assert w.live.values["disk"]["mag"] != mags["disk"]  # re-fitted for the moved disk
    msg = w.statusBar().currentMessage()
    assert msg.startswith("fitted the flux of") and "kept (locked or tied): QSO 1A" in msg
    p._cells["mag"][2].click()  # unlock again
    roi.setPos(roi.pos() - type(roi.pos())(0.05, 0.0))
    p.list.selectAll()
    p.fit_fluxes.click()


@needs_data
def test_settings_switches_flux_units_without_changing_the_model(loaded):
    """F4b: Settings -> fluxes as fluxes / magnitudes converts every flux parameter; chi2 is unchanged."""
    w, _ = loaded
    chi0 = w.result.chi2
    units = {a.data(): a for a in w.flux_units.actions()}
    assert units["mag"].isChecked() and units["mag"].isEnabled()  # the 1A image has a zeropoint
    units["flux"].trigger()
    assert all("mag" not in c["params"] for c in w.spec["components"])
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9) and units["flux"].isChecked()
    sky = w.live.values["sky"]["flux"]
    units["mag"].trigger()  # back; a sky > 0 becomes a surface brightness, a sky <= 0 stays in flux (a note)
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9)
    sky_params = next(c for c in w.spec["components"] if c["name"] == "sky")["params"]
    if sky > 0:
        assert "mag" in sky_params
    else:
        assert "flux" in sky_params and "kept in flux" in w.statusBar().currentMessage()
    p = w.panel
    p.list.setCurrentRow(1)
    assert p.photometry.text().startswith("observed in the box")  # a source-plane component
    p.list.setCurrentRow(3)
    assert p.photometry.text().startswith("intrinsic") and "μ(θ)" in p.photometry.text()  # the quasar
    assert abs(w.setup.points["QSO 1A"].mu) > 1  # it is magnified


@needs_data
def test_add_sersic_fits_its_flux_once_and_remove_restores_the_model(loaded):
    """A3a: "+ Add Sérsic" adds a source-plane ellipse whose flux is the leftover least-squares value
    (once); dragging it keeps that flux; "Remove" drops it and its ties and gives the old chi2 back."""
    from forwardgal.photometry import mag_to_flux

    w, _ = loaded
    p = w.panel
    chi0, colours0, names0 = w.result.chi2, dict(w._colours), list(p.names)
    p.add_button.click()
    name = "sersic1"
    assert p.names == names0 + [name] and p.current == name and name in w.rois
    assert w.rois[name].getViewBox() is w.source.view  # in the source plane
    assert {n: w._colours[n] for n in colours0} == colours0  # the others keep their colours
    assert w._colours[name] not in colours0.values()
    comp = w.spec["components"][-1]
    assert "mag" in comp["params"]  # the Settings unit (magnitudes here)

    # Its flux is the one-component least-squares fit to what the others leave over (or 0 -> faint start).
    lv = w.live
    F = mag_to_flux(lv.values[name]["mag"], lv.zeropoint)
    T = np.asarray(lv.template(name))
    others = w.result.model - F * T
    m, s = lv.mask, lv.sigma
    F_ls = float(np.sum(T[m] * (lv.data[m] - others[m]) / s[m] ** 2) / np.sum(T[m] ** 2 / s[m] ** 2))
    if F_ls > 0:
        assert F == pytest.approx(F_ls, rel=1e-6)
        assert w.result.chi2 < chi0  # it took some leftover light
    mag = lv.values[name]["mag"]
    roi = w.rois[name]
    roi.setPos(roi.pos() + type(roi.pos())(0.03, 0.0))  # dragging it does not refit its flux
    w.render()
    assert lv.values[name]["mag"] == mag

    p.remove_button.click()  # remove it: the model is as before
    assert p.names == names0 and name not in w.rois and name not in w._colours
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9)


@needs_data
def test_removing_a_component_drops_its_ties_and_nothing_else_moves(loaded):
    """A parameter tied to a removed component keeps the value it had (like "untie")."""
    from forwardgal.edit import set_tie

    w, _ = loaded
    n0 = w.live.values["bulge"]["n"]
    name = w.add_sersic("source", -5.7, 0.2, 0.05)
    w.spec = set_tie(w.spec, "bulge.n", f"{name}.n", 0.5)  # bulge.n = sersic.n + 0.5 = 1.5
    w.render()
    assert w.live.values["bulge"]["n"] == pytest.approx(1.5)
    w.panel.select(name)
    w.panel.remove_button.click()
    assert "bulge.n" not in w.spec.get("ties", {}) and "bulge.n" in w.statusBar().currentMessage()
    assert w.live.values["bulge"]["n"] == pytest.approx(1.5)  # kept, not back to its old own value
    from forwardgal.edit import set_param

    w.spec = set_param(w.spec, "bulge", "n", value=n0)  # restore for the other tests
    w.render()


@needs_data
def test_add_sersic_uses_the_settings_unit_and_refuses_bad_shapes(loaded):
    w, _ = loaded
    chi0, names0 = w.result.chi2, list(w.panel.names)
    units = {a.data(): a for a in w.flux_units.actions()}
    units["flux"].trigger()
    name = w.add_sersic("image", 17.0, 3.0, 0.1)  # an image-plane Sersic, by the method the Draw tool will use
    comp = w.spec["components"][-1]
    assert comp["plane"] == "image" and "flux" in comp["params"] and comp["params"]["flux"]["value"] > 0
    assert w.rois[name].getViewBox() is w.image.view
    w.panel.remove_button.click()
    units["mag"].trigger()
    assert w.add_sersic("source", 0.0, 0.0, 5.0) is None  # R_e > 2" in the source plane: refused, said why
    assert "inside the bounds" in w.statusBar().currentMessage() and w.panel.names == names0
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9)


class FakeDrag:
    """Stands in for pyqtgraph's MouseDragEvent: a left-button drag from scene point a to b."""

    def __init__(self, a, b, finish, button=None, view=None):
        from PySide6.QtCore import Qt

        self.a, self.b, self.finish, self.view = a, b, finish, view
        self._button = Qt.MouseButton.LeftButton if button is None else button
        self.accepted = False

    def button(self):
        return self._button

    def buttonDownScenePos(self, button=None):
        return self.a

    def scenePos(self):
        return self.b

    def pos(self):  # (pyqtgraph's own pan handler reads these, in the view's item coordinates)
        return pg.Point(self.view.mapFromScene(self.b))

    def lastPos(self):
        return pg.Point(self.view.mapFromScene(self.a))

    def isFinish(self):
        return self.finish

    def isStart(self):
        return False

    def accept(self):
        self.accepted = True


@needs_data
def test_draw_ellipse_tool_adds_a_sersic_in_the_plane_drawn_in(loaded):
    """A3b: with "Draw ellipse", a drag along the major axis previews and then adds a Sersic there."""
    from PySide6.QtCore import QPointF

    w, _ = loaded
    names0 = list(w.panel.names)
    draw = next(a for a in w.tools.actions() if a.text() == "Draw ellipse")
    assert draw.isEnabled()
    draw.trigger()
    assert w.source.drawing and w.image.drawing

    v = w.source.view
    p0, p1 = (-5.80, 0.10), (-5.60, 0.30)  # beta, arcsec: a drag towards the north-east (PA 45)
    a, b = v.mapViewToScene(QPointF(*p0)), v.mapViewToScene(QPointF(*p1))
    ev = FakeDrag(a, b, finish=False)
    v.mouseDragEvent(ev)  # mid-drag: the dashed preview shows
    assert ev.accepted and w.source.preview.isVisible()
    v.mouseDragEvent(FakeDrag(a, b, finish=True))
    assert not w.source.preview.isVisible()
    name = w.panel.names[-1]
    assert w.panel.names == names0 + [name] and w.panel.current == name
    comp = w.spec["components"][-1]
    assert comp["plane"] == "source"
    val = w.live.values[name]
    assert (val["x0"], val["y0"]) == pytest.approx((-5.70, 0.20), abs=1e-6)
    assert val["R_e"] == pytest.approx(np.hypot(0.2, 0.2) / 2, rel=1e-6)
    from forwardgal.components import sersic_qpa

    q, pa = (float(x) for x in sersic_qpa(val))
    assert q == pytest.approx(0.6, rel=1e-6) and pa == pytest.approx(45.0, abs=1e-6)  # +x east, +y north
    w.panel.remove_button.click()

    # A click (no drag) adds nothing; in the image plane a drag adds an image-plane Sersic.
    vi = w.image.view
    c = vi.mapViewToScene(QPointF(17.0, 3.0))
    vi.mouseDragEvent(FakeDrag(c, c, finish=True))
    assert w.panel.names == names0 and "major axis" in w.statusBar().currentMessage()
    vi.mouseDragEvent(FakeDrag(c, vi.mapViewToScene(QPointF(17.0, 3.4)), finish=True))
    assert w.spec["components"][-1]["plane"] == "image" and w.rois[w.panel.names[-1]].getViewBox() is vi
    w.panel.remove_button.click()

    # Back in Select / move, a drag pans again (the original handler) and draws nothing.
    next(a for a in w.tools.actions() if a.text() == "Select / move").trigger()
    assert not w.source.drawing
    r0 = v.viewRange()
    v.mouseDragEvent(FakeDrag(a, b, finish=False, view=v))
    assert w.panel.names == names0 and not w.source.preview.isVisible()
    assert not np.allclose(v.viewRange(), r0)  # panned
    v.setRange(xRange=r0[0], yRange=r0[1], padding=0)


@needs_data
def test_fit_fluxes_fits_only_the_selected_components(loaded):
    """A4: Ctrl/⌘-click selects several; "Fit fluxes" solves only theirs, the others are kept."""
    from PySide6.QtCore import QItemSelectionModel, Qt

    w, _ = loaded
    p = w.panel
    roi = w.rois["disk"]
    roi.setPos(roi.pos() + type(roi.pos())(0.05, 0.0))  # move the disk: its flux is no longer the best
    w.render()
    before = {n: dict(v) for n, v in w.live.values.items()}
    p.list.setCurrentRow(p.names.index("disk"))  # a plain click: only the disk
    assert p.selected_names() == ["disk"] and p.fit_fluxes.text() == "Fit flux"
    p.list.setCurrentRow(p.names.index("sky"), QItemSelectionModel.Toggle)  # Ctrl/⌘-click the sky
    assert p.selected_names() == ["disk", "sky"] and p.current == "sky" and p.fit_fluxes.text() == "Fit fluxes (2)"
    w.rois["bulge"].sigClicked.emit(w.rois["bulge"], type("E", (), {"modifiers": lambda self: Qt.KeyboardModifier.NoModifier})())
    assert p.selected_names() == ["bulge"]  # a plain click on an ellipse selects only it
    ctrl = type("E", (), {"modifiers": lambda self: Qt.KeyboardModifier.ControlModifier})()
    w.rois["disk"].sigClicked.emit(w.rois["disk"], ctrl)  # Ctrl/⌘-click on an ellipse adds it
    assert p.selected_names() == ["bulge", "disk"] and p.current == "disk"
    assert w.rois["disk"].pen.widthF() == w.rois["bulge"].pen.widthF() == 3.5  # both drawn thick
    w.rois["bulge"].sigClicked.emit(w.rois["bulge"], ctrl)  # ...and again takes it out
    assert p.selected_names() == ["disk"]

    p.fit_fluxes.click()  # only the disk
    after = w.live.values
    assert after["disk"]["mag"] != before["disk"]["mag"]
    for n in ("bulge", "QSO 1A", "sky"):
        assert after[n] == before[n]
    # it is the best disk flux given the others: refitting everything else then the disk changes nothing more
    best = w.live.best_fluxes(["disk"])
    from forwardgal.photometry import mag_to_flux

    assert mag_to_flux(after["disk"]["mag"], w.live.zeropoint) == pytest.approx(best["disk"], rel=1e-9)
    roi.setPos(roi.pos() - type(roi.pos())(0.05, 0.0))
    p.list.selectAll()
    p.fit_fluxes.click()
    p.list.setCurrentRow(0)


@needs_data
def test_selection_survives_adding_a_component(loaded):
    from PySide6.QtCore import QItemSelectionModel

    w, _ = loaded
    p = w.panel
    p.list.setCurrentRow(p.names.index("bulge"))
    p.list.setCurrentRow(p.names.index("disk"), QItemSelectionModel.Toggle)
    name = w.add_sersic("source", -5.7, 0.2, 0.05)
    assert p.selected_names() == [name]  # the new one is selected (to drag it at once)
    p.list.setCurrentRow(p.names.index("bulge"))
    p.list.setCurrentRow(p.names.index("disk"), QItemSelectionModel.Toggle)
    w.render()  # an ordinary re-render keeps the selection
    assert p.selected_names() == ["bulge", "disk"]
    p.select(name)
    p.remove_button.click()
    assert p.names[0] == "bulge" and name not in p.names


@needs_data
def test_no_lens_model_in_the_gui(app, tmp_path):
    """N3: without a lens there is no source plane; "+ Add Sérsic" adds to the image plane; saving keeps it lens-free."""
    from test_no_lens import no_lens_config

    from forwardgal.gui.app import MainWindow, theta_text
    from forwardgal.serialize import read_yaml, write_yaml

    write_yaml(no_lens_config(), tmp_path / "flat.yaml")
    w = MainWindow()
    try:
        assert w.load(tmp_path / "flat.yaml")
        assert not w.setup.lensed and w.source.isHidden()
        assert w.chips["alpha"].text() == "α: none" and w.chips["z_src"].text() == "no lens"
        assert set(w.rois) == {"host"} and w.rois["host"].getViewBox() is w.image.view
        text = theta_text(w.setup, 0.0, 0.0)
        assert "beta" not in text and "RA, Dec" in text
        assert w.panel._notes["QSO 1A"].endswith("(μ(θ) = +1)")  # a point: intrinsic = observed

        w.panel.add_button.click()
        name = w.panel.names[-1]
        assert w.spec["components"][-1]["plane"] == "image" and w.rois[name].getViewBox() is w.image.view
        w.save(tmp_path / "saved.yaml")
        saved = read_yaml(tmp_path / "saved.yaml")
        assert "lens" not in saved and "cosmology" not in saved
        assert [c["plane"] for c in saved["model"]["components"] if c["kind"] == "sersic"] == ["image", "image"]
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


@needs_data
def test_show_and_hide_markers(loaded):
    """V1: the list's narrow boxes hide a component's ellipse or + marker; View -> Hide/Show all. Display only."""
    w, _ = loaded
    p = w.panel
    chi0 = w.result.chi2
    assert p._vis["sky"].isHidden() and not p._vis["disk"].isHidden()  # the sky has no marker
    assert all(r.isVisible() or not w.isVisible() for r in w.rois.values())
    p._vis["disk"].click()  # untick: hide the disk's ellipse
    assert p.hidden == {"disk"} and not w.rois["disk"].isVisible() and w.rois["bulge"].isVisible()
    p._vis["QSO 1A"].click()  # the quasar's + markers
    assert all(not m.isVisible() for m in w._markers["QSO 1A"])
    w.view_actions["show"].trigger()
    assert p.hidden == set() and w.rois["disk"].isVisible() and all(m.isVisible() for m in w._markers["QSO 1A"])
    assert p._vis["disk"].isChecked()
    w.view_actions["hide"].trigger()
    assert not any(r.isVisible() for r in w.rois.values()) and not p._vis["bulge"].isChecked()
    w.render()  # hiding survives a re-render (and a list rebuild when a component is added)
    name = w.add_sersic("source", -5.7, 0.2, 0.05)
    assert w.rois[name].isVisible() and not w.rois["disk"].isVisible()  # a new component is shown
    p.remove_button.click()
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9)  # display only
    w.view_actions["show"].trigger()


@needs_data
def test_grid_toggle(loaded):
    """V2: View -> Grid turns the grid of every plot off and on."""
    w, _ = loaded
    axes = [pl.plot.getAxis(k) for pl in (w.source, w.image, w.image2) for k in ("bottom", "left")]
    assert all(a.grid for a in axes)
    w.view_actions["grid"].trigger()  # off
    assert not w.view_actions["grid"].isChecked() and not any(a.grid for a in axes)
    w.view_actions["grid"].trigger()
    assert all(a.grid for a in axes)


@needs_data
def test_second_image_panel_shows_another_mode_with_linked_pan_and_zoom(loaded):
    """V3: a second image-plane panel (Residual by default) under the first; panning one pans both."""
    w, _ = loaded
    assert w.image2.isHidden()
    w.view_actions["second"].trigger()
    assert not w.image2.isHidden()
    assert w._shown["image2"][0] == "Residual"
    w.mode2.button(1).click()  # Model, with the data's default limits
    np.testing.assert_allclose(w.image2.image.image, default_view(w, w.result.model), atol=1e-12)
    def centre(view):
        (x0, x1), (y0, y1) = view.viewRange()
        return (x0 + x1) / 2, (y0 + y1) / 2

    r0 = w.image.view.viewRange()
    w.image.view.setRange(xRange=(17.0, 18.5), yRange=(3.0, 4.5), padding=0)  # zoom the top panel
    assert centre(w.image2.view) == pytest.approx(centre(w.image.view), abs=1e-9)
    w.image2.view.translateBy(x=0.3, y=-0.2)  # pan the bottom one: the top follows
    assert centre(w.image.view) == pytest.approx(centre(w.image2.view), abs=1e-9)
    assert centre(w.image.view) == pytest.approx((17.75 + 0.3, 3.75 - 0.2), abs=1e-6)
    w.image.view.setRange(xRange=r0[0], yRange=r0[1], padding=0)
    w.mode2.button(2).click()
    w.view_actions["second"].trigger()
    assert w.image2.isHidden()


@needs_data
def test_in_model_boxes_and_data_minus_model_export(loaded, tmp_path):
    """M1: untick "in model" to see the model and residual without a component; Data − model shows the data
    minus the components still in; File -> Export writes it as FITS (box or whole cutout)."""
    from astropy.io import fits

    w, _ = loaded
    p = w.panel
    chi0, mags = w.result.chi2, {n: v.get("mag") for n, v in w.live.values.items()}
    w.panel.set_off(set(p.names) - {"QSO 1A"})  # only the quasar left in the model
    assert w.live.off == {"bulge", "disk", "sky"} and not p._inmodel["disk"].isChecked()
    assert "without bulge, disk, sky" in w.fitinfo.text() and w.result.chi2 > chi0
    T = np.asarray(w.live.template("QSO 1A"))
    np.testing.assert_allclose(w.result.model, w.live.amplitude("QSO 1A") * T, rtol=1e-12)  # just the quasar
    from PySide6.QtCore import Qt

    assert w.rois["disk"].pen.style() == Qt.DashLine
    assert p._notes["disk"].startswith("not in the model now")
    w.mode.button(3).click()  # Data − model: the quasar-subtracted data
    np.testing.assert_allclose(w.image.image.image, default_view(w, w.live.data - w.result.model), atol=1e-12)

    # Fit fluxes leaves "off" components out (never changes their flux).
    p.list.selectAll()
    p.fit_fluxes.click()
    assert all(w.live.values[n].get("mag") == mags[n] for n in ("bulge", "disk"))
    assert "left out (not in the model): bulge, disk, sky" in w.statusBar().currentMessage()

    sub = w.export_fits(tmp_path / "box.fits", "box")
    assert sub == ["QSO 1A"]
    with fits.open(tmp_path / "box.fits") as h:
        np.testing.assert_allclose(h[0].data, (w.live.data - w.result.model).astype(np.float32))
        assert h[0].header["SUBTR1"] == "QSO 1A" and h[0].header["BUNIT"] == w.setup.img.bunit
    w.export_fits(tmp_path / "cut.fits", "cutout")
    with fits.open(tmp_path / "cut.fits") as h:
        assert h[0].data.shape == w.setup.img.data.shape

    w.panel.set_off(set())  # back: everything in, as before (the quasar flux may have been refitted)
    w.mode.button(0).click()
    assert w.live.off == set() and "without" not in w.fitinfo.text()
    p.list.selectAll()
    p.fit_fluxes.click()


@needs_data
def test_locking_q_or_pa_turns_an_e1e2_sersic_into_q_pa(loaded):
    """G2e: locking PA of the (e1, e2) disk converts it to (q, PA) at the same shape, then locks PA."""
    from forwardgal.components import sersic_qpa

    w, _ = loaded
    p = w.panel
    spec0, chi0, dof0 = w.spec, w.result.chi2, w.result.dof
    q0, pa0 = (float(x) for x in sersic_qpa(w.live.values["disk"]))
    p.select("disk")
    p._cells["pa_deg"][2].click()  # lock PA
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    if "e1" in disk["params"]:
        assert "to_qpa" in w.statusBar().currentMessage()
        pytest.skip("waiting for Erik's edit.to_qpa")
    assert {"q", "pa_deg"} <= set(disk["params"]) and disk["params"]["pa_deg"]["free"] is False
    assert w.live.values["disk"]["q"] == pytest.approx(q0, abs=1e-9)
    assert w.live.values["disk"]["pa_deg"] == pytest.approx(pa0 % 180, abs=1e-6)
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9) and w.result.dof == dof0 + 1  # same model, one fewer
    assert "now in (q, PA)" in w.statusBar().currentMessage() and p._cells["pa_deg"][2].isChecked()
    assert p._cells["q"][1].text() == "U(0.05, 1)"  # q has bounds of its own now
    p._cells["q"][1].click()  # the constraint dialog of q opens directly (already converted)
    assert w.dialog is not None
    w.dialog.reject()
    w.spec = spec0
    w.render()


@needs_data
def test_history_table_save_load_trash_restore(app, tmp_path):
    """G3: Save to history makes numbered drafts (started_from chains them); Load asks for a fit (best-fit or
    starting values) and puts the model back; trash moves a folder to .trash, restore brings it back."""
    import shutil
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.gui.app import MainWindow
    from forwardgal.serialize import read_yaml, write_yaml

    write_yaml(example_config(), tmp_path / "model.yaml")
    w = MainWindow()
    try:
        assert w.load(tmp_path / "model.yaml")
        h = w.history
        assert w.history_dir == tmp_path / "fits" and h.rows == [] and h.trash_button.text() == "Trash (0)"
        h.save_button.click()
        d1 = "0001_1A-bulge-disk"
        assert [r["folder"] for r in h.rows] == [d1] and w.loaded_from == d1 and h.loaded == d1
        assert h.rows[0]["status"] == "unfitted" and h.rows[0]["chi2_per_dof"] == pytest.approx(w.result.chi2 / w.result.dof)
        assert not h.button(d1, "load").isEnabled() and h.button(d1, "load").text() == "Loaded"
        saved = read_yaml(tmp_path / "fits" / d1 / "model.yaml")
        assert Path(saved["psf"]["file"]).is_absolute()

        n0 = w.live.values["disk"]["n"]
        from forwardgal.edit import set_param

        w.spec = set_param(w.spec, "disk", "n", value=2.0)  # an edit, then a second draft from the first
        w.render()
        assert w.unsaved()
        f2 = w.save_to_history()
        assert f2.name == "0002_1A-bulge-disk" and h.rows[1]["started_from"] == d1 and not w.unsaved()

        # Load the first draft (no question: nothing unsaved) -> its spec (n back to the old value).
        h.button(d1, "load").click()
        assert w.loaded_from == d1 and w.live.values["disk"]["n"] == pytest.approx(n0)

        # A fit folder (made by hand here: model.yaml = start, bestfit.yaml = other values, result.yaml).
        fit = tmp_path / "fits" / "0003_1A-bulge-disk"
        shutil.copytree(tmp_path / "fits" / d1, fit)
        (fit / "draft.yaml").unlink()
        best = read_yaml(fit / "model.yaml")
        next(c for c in best["model"]["components"] if c["name"] == "disk")["params"]["n"]["value"] = 1.7
        write_yaml(best, fit / "bestfit.yaml")
        write_yaml({"label": "1A-bulge-disk", "created": "2026-11-23T12:00:00", "chi2": 1.0e4, "dof": 10039,
                    "chi2_per_dof": 0.996, "n_pixels": 10046, "started_from": d1,
                    "starts": [{"loss": 1.0, "iterations": 5, "converged": True}]}, fit / "result.yaml")
        w.refresh_history()  # (the watcher would do this in the running app)
        assert h.rows[2]["kind"] == "fit" and h.rows[2]["status"] == "optimizer"
        h.button(fit.name, "load").click()  # a fit: it asks
        box = w.question
        assert box is not None and "best-fit" in box.text()
        best_button = next(b for b in box.buttons() if b.text() == "Best-fit values")
        best_button.click()
        assert w.loaded_from == fit.name and w.live.values["disk"]["n"] == pytest.approx(1.7)
        w.load_history(fit.name, "start")
        assert w.live.values["disk"]["n"] == pytest.approx(n0)

        # Unsaved changes: even a draft asks first; Cancel keeps the current model.
        w.spec = set_param(w.spec, "disk", "n", value=3.0)
        w.render()
        h.button(d1, "load").click()
        assert "unsaved changes" in w.question.text()
        next(b for b in w.question.buttons() if w.question.buttonRole(b) == w.question.ButtonRole.RejectRole).click()
        assert w.live.values["disk"]["n"] == pytest.approx(3.0)

        h.button(f2.name, "trash").click()
        assert [r["folder"] for r in h.rows] == [d1, fit.name] and h.trash_button.text() == "Trash (1)"
        h.trash_menu.actions()[0].trigger()  # restore it
        assert [r["folder"] for r in h.rows] == [d1, f2.name, fit.name] and h.trash_button.text() == "Trash (0)"
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


@needs_data
def test_fit_from_the_gui_runs_in_a_process_and_loads_the_best_fit(app, tmp_path):
    """G4: Fit selected saves a draft, runs forwardgal.run --only in a QProcess, and loads its best fit."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.gui.app import MainWindow
    from forwardgal.serialize import read_yaml, write_yaml

    cfg = example_config()
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 3}
    write_yaml(cfg, tmp_path / "model.yaml")
    w = MainWindow()
    try:
        assert w.load(tmp_path / "model.yaml")
        assert w.fitpanel.starts.value() == 1 and not w.fitpanel.cancel.isEnabled()
        w.panel.set_off({"sky"})  # refused while something is "off"
        assert not w.start_fit() and "in model" in w.statusBar().currentMessage()
        w.panel.set_off(set())
        bulge0 = dict(w.live.values["bulge"])
        w.panel.select("disk")
        w.fitpanel.fit_selected.click()
        assert w.process is not None and not w.fitpanel.fit_all.isEnabled() and w.fitpanel.cancel.isEnabled()
        assert w.history.rows[0]["kind"] == "draft"  # saved first
        assert w.process.waitForFinished(300_000)
        app.processEvents()
        assert w.process is None and "best fit loaded" in w.fitpanel.progress.text(), w._fit_log[-5:]
        fit = w.history.rows[-1]
        assert fit["kind"] == "fit" and fit["started_from"] == w.history.rows[0]["folder"]
        assert fit["status"].endswith("only disk") and w.loaded_from == fit["folder"]
        assert w.live.values["bulge"] == pytest.approx(bulge0)  # held
        assert read_yaml(tmp_path / "fits" / fit["folder"] / "result.yaml")["fitted_only"] == ["disk"]

        w.fitpanel.fit_all.click()  # and Cancel
        assert w.process is not None
        w.fitpanel.cancel.click()
        if w.process is not None:
            w.process.waitForFinished(30_000)
        app.processEvents()
        assert w.process is None and "cancelled" in w.fitpanel.progress.text()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


@needs_data
def test_nuts_from_the_gui(app, tmp_path):
    """Run NUTS -> posteriors: only with an unchanged fit loaded; runs forwardgal.posterior in a QProcess; the NUTS
    row is loaded and its corner plot can be opened."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.gui.app import MainWindow
    from forwardgal.run import run
    from forwardgal.serialize import write_yaml

    cfg = example_config()
    cfg["fit"] = {"method": "lm", "n_starts": 1, "seed": 0, "max_iter": 3}
    write_yaml(cfg, tmp_path / "model.yaml")
    fit_folder, *_ = run(tmp_path / "model.yaml", errors=False, plot=False, progress=False, only=["disk"])
    w = MainWindow()
    try:
        assert w.load(tmp_path / "model.yaml")
        assert not w.fitpanel.nuts.isEnabled() and "load a fit" in w.fitpanel.nuts.toolTip()
        w.load_history(fit_folder.name, "bestfit")
        assert w.fitpanel.nuts.isEnabled()
        w.fitpanel.warmup.setValue(10)
        w.fitpanel.samples.setValue(10)
        w.fitpanel.chains.setValue(1)
        w.fitpanel.nuts.click()
        assert w.process is not None and not w.fitpanel.nuts.isEnabled()
        assert w.process.waitForFinished(600_000)
        app.processEvents()
        assert w.process is None and "posteriors" in w.fitpanel.progress.text(), w._fit_log[-5:]
        row = w.history.rows[-1]
        assert row["status"].startswith("NUTS") and row["started_from"] == fit_folder.name
        assert w.loaded_from == row["folder"] and w.fitpanel.corner.isEnabled() and w._corner.exists()
        assert w.fitpanel.nuts.isEnabled()  # a NUTS folder is a fit too: NUTS again from it is allowed
        from forwardgal.edit import set_param

        w.spec = set_param(w.spec, "disk", "n", value=1.5)  # changed: NUTS would not start at the saved fit
        w.render()
        assert not w.fitpanel.nuts.isEnabled() and "changed" in w.fitpanel.nuts.toolTip()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()



@needs_data
def test_scale_menu_per_display_and_back_to_default(loaded):
    """D3: Scale -> a display -> scale / limits; only that display changes; Default scaling restores it."""
    from forwardgal.scaling import stretch

    w, _ = loaded
    w.mode.button(0).click()  # Data on top
    before = np.array(w.image.image.image)
    np.testing.assert_allclose(before, default_view(w, w.live.data), atol=1e-12)  # the default = the old look
    scale_g, lim_g = w.scale_groups["Data"]
    next(a for a in lim_g.actions() if a.data() == "minmax").trigger()
    next(a for a in scale_g.actions() if a.data() == "sqrt").trigger()
    d = w.live.data
    lo, hi = d[w.live.mask].min(), d[w.live.mask].max()
    assert w._shown["image"] == ("Data", pytest.approx((lo, hi)))
    np.testing.assert_allclose(w.image.image.image, stretch(np.clip((d - lo) / (hi - lo), 0, 1), "sqrt"), atol=1e-12)
    assert w.scales["Model"].scale == "default"  # only Data changed
    w.set_scale("Data", contrast=2.0, bias=0.4)
    assert w.scales["Data"].contrast == 2.0
    w.set_scale("Data", limits="user")  # without lo, hi: refused, said why
    assert w.scales["Data"].limits == "minmax" and "lo < hi" in w.statusBar().currentMessage()
    w.reset_scale("Data")
    np.testing.assert_allclose(w.image.image.image, before, atol=1e-12)
    assert next(a for a in scale_g.actions() if a.isChecked()).data() == "default"
    w.set_scale("Source plane", scale="log", limits="99")
    w.reset_scale(None)  # All to default
    assert all(sc == w.scales["Data"] for sc in w.scales.values())


@needs_data
def test_caustics_toggle(loaded):
    w, _ = loaded
    assert w._curve_items and all(i.isVisible() or not w.isVisible() for i in w._curve_items)
    w.view_actions["curves"].trigger()
    assert not any(i.isVisible() for i in w._curve_items)
    w.view_actions["curves"].trigger()
    assert w.view_actions["curves"].isChecked()


@needs_data
def test_colorbar_drag_changes_contrast_and_bias_of_the_shown_display(loaded):
    """D4: dragging the colorbar under a panel sets contrast / bias of the display it shows (DS9-like)."""
    from forwardgal.scaling import colormap_position

    w, _ = loaded
    w.mode.button(0).click()  # Data
    base = np.array(w.image.image.image)
    bar = w.image.colorbar
    bar.drag_to(1.0, 0.5, dx=-0.1 * bar.width(), dy=-60.0)  # left: bias + 0.1; up 60 px: contrast x e
    sc = w.scales["Data"]
    assert sc.bias == pytest.approx(0.6) and sc.contrast == pytest.approx(np.e)
    assert w.scales["Model"].contrast == 1.0  # only the shown display
    np.testing.assert_allclose(w.image.image.image, colormap_position(base, np.e, 0.6), atol=1e-12)
    assert (bar.contrast, bar.bias) == pytest.approx((np.e, 0.6))
    w.image.colorbar_reset.click()  # Default
    assert (w.scales["Data"].contrast, w.scales["Data"].bias) == (1.0, 0.5)
    np.testing.assert_allclose(w.image.image.image, base, atol=1e-12)


@needs_data
def test_scale_parameters_dialog_sets_user_limits(loaded):
    """D5: Scale Parameters shows the histogram with the current limits (data units); dragging the band and
    Apply sets user limits; a bad pair is refused in the dialog."""
    w, _ = loaded
    w.mode.button(0).click()  # Data
    w.open_scale_parameters("Data")
    dlg = w.dialog
    lo0, hi0 = dlg.region.getRegion()  # the default limits, back in data units
    t = np.arcsinh(w.live.data[w.live.mask] / w._soft)
    assert (lo0, hi0) == pytest.approx(tuple(w._soft * np.sinh(np.percentile(t, [0.5, 99.7]))), rel=1e-6)
    dlg.region.setRegion((0.0, 0.5))  # "drag" the band
    assert float(dlg.lo.text()) == 0.0 and float(dlg.hi.text()) == 0.5
    dlg.apply_button.click()
    sc = w.scales["Data"]
    assert (sc.limits, sc.lo, sc.hi) == ("user", 0.0, 0.5) and w._shown["image"][1] == pytest.approx(
        (0.0, np.arcsinh(0.5 / w._soft)))  # (the default asinh scale, between the user limits)
    lim_g = w.scale_groups["Data"][1]
    assert next(a for a in lim_g.actions() if a.isChecked()).data() == "user"
    dlg.lo.setText("1.0")
    dlg.hi.setText("0.2")
    dlg.apply_button.click()
    assert "below" in dlg.error.text() and w.scales["Data"].hi == 0.5  # refused, unchanged
    dlg.reject()
    w.reset_scale("Data")


@needs_data
def test_bands_tabs_switching_and_use_in_band(app, tmp_path):
    """G5: one tab per band (named by FILTER); only the active band is built and edits survive switching;
    "Use in band" seeds the other band from a fit (a draft in ITS history, started_from "F115W:<fit>")."""
    import shutil
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    from forwardgal.config import upgrade_config
    from forwardgal.edit import set_param
    from forwardgal.gui.app import MainWindow
    from forwardgal.seed import DEFAULT_RANGES
    from forwardgal.serialize import read_yaml, write_yaml

    fit = ROOT / "examples" / "fits" / "0001_1A-bulge-disk"
    if not fit.exists():
        pytest.skip("needs the F115W example fit")
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    write_yaml(example_config(), tmp_path / "A" / "model.yaml")
    shutil.copytree(fit, tmp_path / "A" / "fits" / fit.name)
    b = upgrade_config(example_config())
    b["data"]["image"]["file"] = b["data"]["sigma"]["file"] = str(
        ROOT / "data" / "jw06675-o001_t002_nircam_clear-f150w_i2d.fits")
    b["psf"]["file"] = str(ROOT / "data" / "rotated_jw06675-o001_t002_nircam_clear-f150w-QSOblueA_PSF_i2dtrim.fits")
    b["label"] = "1A-F150W"
    next(c for c in b["model"]["components"] if c["kind"] == "point")["max_snap"] = None
    write_yaml(b, tmp_path / "B" / "model.yaml")

    w = MainWindow()
    try:
        assert w.load(tmp_path / "A" / "model.yaml")
        assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["F115W"]
        assert w.add_band(tmp_path / "B" / "model.yaml")
        assert w.active == 1 and w.tabs.tabText(1) == "F150W" and w.history_dir == tmp_path / "B" / "fits"
        w.spec = set_param(w.spec, "disk", "n", value=1.7)  # an unsaved edit in F150W
        w.render()
        w.tabs.setCurrentIndex(0)  # -> F115W, built again
        assert w.active == 0 and "f115w" in w.cfg["data"]["image"]["file"] and w.history.rows
        w.tabs.setCurrentIndex(1)  # back: the edit is still there
        assert w.live.values["disk"]["n"] == pytest.approx(1.7) and w.unsaved()

        w.tabs.setCurrentIndex(0)
        assert w.history.button(fit.name, "use").isVisibleTo(w.history.table)  # a fit row has "Use in band…"
        w.history.button(fit.name, "use").click()  # the dialog: F150W is the only other band
        dlg = w.dialog
        assert dlg.target.itemText(0) == "F150W" and dlg.fields["R_e"][0].text() == "20"
        assert dlg.ranges() == DEFAULT_RANGES
        dlg.ok()  # seed F150W from the F115W fit
        assert w.active == 1 and w.history_dir == tmp_path / "B" / "fits"
        row = next(r for r in w.history.rows if r["folder"] == w.loaded_from)
        assert row["kind"] == "draft" and row["started_from"] == f"F115W:{fit.name}"
        best = read_yaml(fit / "result.yaml")["values"]
        assert w.live.values["disk"]["R_e"] == pytest.approx(best["disk"]["R_e"], rel=1e-6)  # A's shapes
        assert "f150w" in w.cfg["data"]["image"]["file"]  # B's set-up
        disk = next(c for c in w.spec["components"] if c["name"] == "disk")
        assert "mag" in disk["params"]  # fluxes from one linear solve in B
        assert w.tabs.tabText(1).startswith("F150W · ")
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


@needs_data
def test_hover_shows_the_pixel_value_of_the_panel(loaded):
    """The read-out adds the value under the mouse of what that panel shows (with units)."""
    w, _ = loaded
    w.mode.button(0).click()  # top: Data
    r, c = 40, 30
    from forwardgal.gui.geometry import pixel_to_theta

    A, t, _ = pixel_to_theta(w.live.grid)
    tx, ty = A @ np.array([c, r]) + t  # theta of pixel [r, c]'s centre
    assert w.pixel_at(tx, ty) == (r, c)
    w._image_mouse(tx, ty, "image")
    assert f"Data = {w.live.data[r, c]:.5g} {w.setup.img.bunit}" in w.coords.text()
    w.view_actions["second"].trigger()  # bottom: Residual, in sigma
    w._image_mouse(tx, ty, "image2")
    res = w.result.residual[r, c]
    assert (f"Residual = {res:.5g} σ" if np.isfinite(res) else "outside the mask") in w.coords.text()
    w.view_actions["second"].trigger()


@needs_data
def test_smooth_menu_per_display(loaded):
    """Smooth -> a display -> Smooth (on), function, Smooth Parameters; only that display is smoothed."""
    from forwardgal.smoothing import Smooth, smooth

    w, _ = loaded
    w.mode.button(2).click()  # Residual on top
    on, funcs = w.smooth_actions["Residual"]
    on.trigger()
    assert w.smooths["Residual"].on and on.isChecked() and not w.smooths["Data"].on
    mode, raw, shown = w._arrays["image"]
    assert mode == "Residual"
    np.testing.assert_allclose(shown, smooth(w.result.residual, Smooth(on=True)), equal_nan=True)
    assert np.all(np.isnan(shown[~w.live.mask]))  # outside the mask: still no value (not smeared in)
    next(a for a in funcs.actions() if a.data() == "gaussian").trigger()
    assert w.smooths["Residual"].function == "gaussian"
    w.open_smooth_parameters("Residual")
    w.dialog.radius.setValue(5)
    w.dialog.sigma.setText("1.5")
    w.dialog.apply_button.click()
    assert w.smooths["Residual"] == Smooth(on=True, function="gaussian", radius=5, sigma=1.5)
    w.dialog.sigma.setText("-1")
    w.dialog.apply_button.click()
    assert "sigma" in w.dialog.error.text()  # refused, unchanged
    w.dialog.reject()
    on.trigger()  # off again
    np.testing.assert_allclose(w._arrays["image"][2], w.result.residual, equal_nan=True)
    w.mode.button(0).click()


@needs_data
def test_add_edit_and_remove_modes_in_the_card(loaded):
    """G6a: "+ mode" adds a Fourier / bending mode at zero amplitude (same chi2); a_m, phi_m are typed in the
    card (edit.set_mode); "− mode" removes it again."""
    w, _ = loaded
    p = w.panel
    p.select("disk")
    chi0, size0 = w.result.chi2, w.live.params.size
    assert p.add_mode_button.isVisible() or not w.isVisible()
    labels = [a.text() for a in p.add_mode_menu.actions()]
    assert "Fourier m = 1" in labels and "bending m = 5" in labels and not p.remove_mode_button.isEnabled()
    next(a for a in p.add_mode_menu.actions() if a.text() == "Fourier m = 3").trigger()
    assert w.live.params.size == size0 + 2 and w.result.chi2 == pytest.approx(chi0, rel=1e-9)  # zero amplitude
    assert "a3" in p._cells and "phi3" in p._cells and p.remove_mode_button.isEnabled()
    next(a for a in p.add_mode_menu.actions() if a.text() == "bending m = 2").trigger()
    # (a bent profile is integrated in its own bend frame (M12f), another numerical path: at b = 0 it agrees with
    # the plain Sersic to ~1e-5, not bit for bit)
    assert "b2" in p._cells and w.result.chi2 == pytest.approx(chi0, rel=1e-4)

    value = p._cells["a3"][0]  # type a_3 = 0.1 (needs Erik's set_mode)
    value.setText("0.1")
    value.editingFinished.emit()
    disk = next(c for c in w.spec["components"] if c["name"] == "disk")
    if disk["params"]["c3"]["value"] == 0.0:
        assert "set_mode" in w.statusBar().currentMessage()
    else:
        assert float(p._cells["a3"][0].text()) == pytest.approx(0.1, abs=1e-4) and w.result.chi2 != chi0

    for label in ("Fourier m = 3", "bending m = 2"):
        next(a for a in p.remove_mode_menu.actions() if a.text() == label).trigger()
    assert w.live.params.size == size0 and w.result.chi2 == pytest.approx(chi0, rel=1e-9)
    assert "a3" not in p._cells and not p.remove_mode_button.isEnabled()


@needs_data
def test_bending_triangles(loaded):
    """A bending mode gets a draggable triangle on its ridge line; dragging it sideways sets b_m; it follows
    the ellipse, hides with the marker box, and goes with the mode."""
    from PySide6.QtCore import QPointF

    from forwardgal.components import sersic_qpa
    from forwardgal.gui.geometry import bend_handle_pos

    w, _ = loaded
    p = w.panel
    p.select("disk")
    next(a for a in p.add_mode_menu.actions() if a.text() == "bending m = 3").trigger()
    h = w.bend_handles["disk"][3]
    assert h.getViewBox() is w.source.view  # the disk is a source-plane component
    v = w.live.values["disk"]
    pa, ref = float(sersic_qpa(v)[1]), w.live.components["disk"].bend_pa_ref
    assert (h.pos().x(), h.pos().y()) == pytest.approx(bend_handle_pos(v["x0"], v["y0"], v["R_e"], pa, ref, 3, 0.0))
    target = bend_handle_pos(v["x0"], v["y0"], v["R_e"], pa, ref, 3, 0.02)  # where b3 = 0.02 puts it
    off = QPointF(0.05 * np.sin(np.radians(pa)), 0.05 * np.cos(np.radians(pa)))  # + some drag along the axis
    h.setPos(QPointF(*target) + off)  # "drag" (sigPositionChanged)
    w.render()
    assert w.live.values["disk"]["b3"] == pytest.approx(0.02, abs=1e-9)
    assert (h.pos().x(), h.pos().y()) == pytest.approx(target, abs=1e-9)  # put back onto its line
    roi = w.rois["disk"]
    roi.setPos(roi.pos() + type(roi.pos())(0.03, 0.0))  # move the ellipse: the triangle follows
    w.render()
    v = w.live.values["disk"]
    assert (h.pos().x(), h.pos().y()) == pytest.approx(
        bend_handle_pos(v["x0"], v["y0"], v["R_e"], float(sersic_qpa(v)[1]), ref, 3, 0.02), abs=1e-9)
    p._vis["disk"].click()  # hide the disk's markers: the triangle too
    assert not h.isVisible()
    p._vis["disk"].click()
    next(a for a in p.remove_mode_menu.actions() if a.text() == "bending m = 3").trigger()
    assert "disk" not in w.bend_handles and h.getViewBox() is None
    roi.setPos(roi.pos() - type(roi.pos())(0.03, 0.0))
    w.render()


@needs_data
def test_model_settings_and_z_src(loaded):
    """G6b: Settings -> Model settings… changes render / fit / cosmology sections and rebuilds with the same
    model; z_src in the header rescales the deflections (the chip shows the factor); an edit counts as unsaved."""
    from forwardgal.cosmology import Cosmology, deflection_scale

    w, _ = loaded
    spec0, chi0, cfg0 = w.spec, w.result.chi2, w.cfg
    assert w.z_src.text() == "1.524" and w.chips["scale"].text() == "α × 1.000"
    w.open_settings()
    dlg = w.dialog
    assert dlg.render["subsample"].value() == 5 and dlg.margin.value() == 35
    assert dlg.sections() == {}  # nothing changed yet
    dlg.render["subsample"].setValue(3)
    dlg.method.setCurrentText("lbfgs")
    assert set(dlg.sections()) == {"render", "fit"}
    dlg.render["patch"].setValue(6)
    dlg.ok()
    assert "odd" in dlg.error.text()  # refused, said why; still open
    dlg.render["patch"].setValue(7)
    dlg.ok()
    assert w.cfg["render"]["subsample"] == 3 and w.cfg["fit"]["method"] == "lbfgs"
    assert w.live.settings.subsample == 3 and w.unsaved()
    assert w.spec == spec0 and w.result.chi2 != chi0  # the same model, rendered with 3 x 3 points

    w.z_src.setText("2.0")  # the source redshift: the maps are rescaled
    w.z_src.editingFinished.emit()
    c = w.cfg["cosmology"]
    f = deflection_scale(c["z_lens"], w.cfg["lens"]["z_ref"], 2.0, Cosmology(c["H0"], c["Om0"], c["Ode0"]))
    assert c["z_src"] == 2.0 and w.chips["scale"].text() == f"α × {f:.3f}" and f > 1
    w.z_src.setText("0.3")  # in front of the lens: refused
    w.z_src.editingFinished.emit()
    assert w.cfg["cosmology"]["z_src"] == 2.0 and "z_lens" in w.statusBar().currentMessage()

    back = {k: cfg0[k] for k in ("render", "fit", "cosmology") if k in cfg0}
    back.setdefault("render", {})
    back.setdefault("fit", {})
    assert w.apply_settings(back) is None
    assert w.result.chi2 == pytest.approx(chi0, rel=1e-9) and w.chips["scale"].text() == "α × 1.000"


@needs_data
def test_edit_mask_tool(loaded):
    """G6c: Edit mask shows the mask polygon (theta, from its RA/Dec); a moved corner + Apply gives a new
    polygon_radec, a rebuilt render box and more fit pixels, with the same model; Revert drops edits."""
    from PySide6.QtCore import QPointF

    w, _ = loaded
    spec0, n0, radec0 = w.spec, int(w.live.mask.sum()), w.cfg["mask"]["polygon_radec"]
    edit = next(a for a in w.tools.actions() if a.text() == "Edit mask")
    assert edit.isEnabled()
    edit.trigger()
    assert w.mask_roi is not None and w.apply_mask_button.isVisibleTo(w.image)
    v = w.mask_vertices()
    np.testing.assert_allclose(v, w.mask_theta(), atol=1e-9)  # the current mask, in theta
    assert len(v) == len(radec0)

    # pull the corner farthest from the centre outward by 0.5"
    c = v.mean(axis=0)
    k = int(np.argmax(np.hypot(*(v - c).T)))
    out = (v[k] - c) / np.hypot(*(v[k] - c))
    w.mask_roi.movePoint(k, QPointF(*(v[k] + 0.5 * out)))
    assert np.hypot(*(w.mask_vertices()[k] - v[k])) == pytest.approx(0.5, abs=1e-6)
    w.revert_mask_button.click()  # Revert: back to the current mask
    np.testing.assert_allclose(w.mask_vertices(), w.mask_theta(), atol=1e-9)

    w.mask_roi.movePoint(k, QPointF(*(v[k] + 0.5 * out)))
    w.apply_mask_button.click()
    assert w.cfg["mask"]["polygon_radec"] != radec0 and int(w.live.mask.sum()) > n0
    assert w.spec == spec0 and w.unsaved()  # the same model; the mask is an unsaved change
    np.testing.assert_allclose(w.mask_theta()[k], v[k] + 0.5 * out, atol=1e-6)  # theta -> RA/Dec -> theta

    w.apply_settings({"mask": dict(w.cfg["mask"], polygon_radec=radec0)})  # back for the other tests
    assert int(w.live.mask.sum()) == n0
    next(a for a in w.tools.actions() if a.text() == "Select / move").trigger()
    if w.mask_roi is not None:  # hidden outside the Edit mask tool
        assert not w.mask_roi.isVisible()


@needs_data
def test_point_options(loaded):
    """G6d: the quasar's options: no snap limit, a PSF of its own with a reference, and a moved position
    (RA/Dec), each rebuilt; its flux and the other components stay; ra/dec is what gets saved."""
    from forwardgal.config import restore_radec

    w, _ = loaded
    p = w.panel
    p.select("QSO 1A")
    assert not p.point_button.isHidden() and p.add_mode_button.isHidden()
    spec0, cfg0 = w.spec, w.cfg
    mag0, disk0 = w.live.values["QSO 1A"]["mag"], dict(w.live.values["disk"])
    p.point_button.click()
    dlg = w.dialog
    info = w.setup.points["QSO 1A"]
    assert float(dlg.ra.text()) == pytest.approx(info.radec[0], abs=1e-8)
    dlg.no_limit.setChecked(True)  # trust the PSF
    dlg.use_own.setChecked(True)
    dlg.file.setText(str(w.setup.files["psf"]))
    dlg.reference.kind.setCurrentText("centre")
    dlg.ok()
    q = next(c for c in restore_radec(w.spec, w.setup.radec)["components"] if c["name"] == "QSO 1A")
    assert q["max_snap"] is None and q["psf"]["reference"] == "centre" and "ra" in q  # saved by RA/Dec
    assert w.live.values["QSO 1A"]["mag"] == mag0 and w.setup.points["QSO 1A"].psf.endswith(".fits")
    assert w.live.values["disk"] == pytest.approx(disk0)  # the other components untouched

    p.select("QSO 1A")
    p.point_button.click()
    dlg = w.dialog
    dec = float(dlg.dec.text())
    dlg.dec.setText(f"{dec + 0.05 / 3600:.10f}")  # 0.05" north
    dlg.use_global.setChecked(True)
    dlg.ok()
    assert w.setup.points["QSO 1A"].theta[1] == pytest.approx(info.theta[1] + 0.05, abs=1e-3)
    assert w.unsaved()

    w.apply_settings({"psf": cfg0["psf"]}, model=restore_radec(spec0, w.setup.radec))  # back
    q = next(c for c in w.spec["components"] if c["name"] == "QSO 1A")
    assert q["params"]["y0"]["value"] == pytest.approx(info.theta[1], abs=1e-6)


@needs_data
def test_window_content_scrolls_on_a_small_screen(loaded):
    """The content has a minimum size; a smaller window scrolls instead of squeezing it."""
    from forwardgal.gui.app import MIN_CONTENT

    w, _ = loaded
    assert w.centralWidget() is w.scroll and w.scroll.widgetResizable()
    inner = w.scroll.widget()
    assert (inner.minimumWidth(), inner.minimumHeight()) == MIN_CONTENT
