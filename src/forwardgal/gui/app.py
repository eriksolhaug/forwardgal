"""forwardgal main window (G1b): source and image planes with live rendering.

Layout after the mockup (outputs/gui_mockup.jpeg): a header with the input
files and tool modes; the source plane (beta) with one draggable ellipse per
source-plane Sersic and the caustics; the image plane (theta) with
Data / Model / Residual, the mask, the critical curves and the quasar; the
components column (G2b: list + parameter card, gui/components_panel.py); the
mouse position and chi^2 in the status bar.

The GUI is a thin layer over the model spec (the ``model:`` section of the
YAML): dragging an ellipse edits the spec (live.set_ellipse), and a 30 ms
timer renders the latest spec with live.LiveModel (the fitter's own template
code). Nothing physical is computed here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QFileSystemWatcher, QProcess, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from forwardgal.components import sersic_qpa
from forwardgal.config import absolute_paths, band_name, build, restore_radec
from forwardgal.critical import caustics, critical_curves
from forwardgal.edit import (add_mode, add_sersic, check, new_name, remove_component, remove_mode, set_mode,
                             set_param, set_tie, tie_candidates, to_qpa)
from forwardgal.gui.components_panel import ComponentsPanel
from forwardgal.gui.fit_panel import FitPanel
from forwardgal.gui.history_panel import HistoryPanel
from forwardgal.history import FOLDER_RE, history_dir_for, list_history, list_trash, restore, save_draft, trash
from forwardgal.gui.constraint_dialog import ConstraintDialog
from forwardgal.gui.geometry import bend_from_handle, bend_handle_pos, drag_to_ellipse, ellipse_to_roi, image_qtransform, pixel_to_theta, roi_to_ellipse
from forwardgal.gui.planes import COMPONENT_COLOURS, GOLD, TEAL, PlanePanel
from forwardgal.gui.rows import param_rows, parse_value
from forwardgal.gui.tie_dialog import TieDialog
from forwardgal.live import LiveModel, set_ellipse
from forwardgal.photometry import add_flux_parameters, convert_fluxes, flux_to_mag, set_fluxes
from forwardgal.scaling import Default, Scale, apply
from forwardgal.smoothing import FUNCTIONS, Smooth, smooth
from forwardgal.serialize import read_yaml, write_yaml

STYLE = """
QMainWindow, QWidget { background: #14161a; color: #d6d9de; font-size: 13px; }
QWidget#card { background: #181b20; border: 1px solid #2a2e35; border-radius: 8px; }
QToolBar { background: #14161a; border: none; spacing: 6px; padding: 4px; }
QLabel#chip { background: #22262d; border-radius: 5px; padding: 3px 8px; font-family: Menlo, monospace; }
QPushButton, QToolButton { background: #22262d; border: 1px solid #343944; border-radius: 6px; padding: 4px 10px; }
QPushButton:checked, QToolButton:checked { background: #f0a04b; color: #14161a; border-color: #f0a04b; }
QPushButton:disabled, QToolButton:disabled { color: #5d636d; }
QStatusBar { background: #14161a; color: #a5abb4; }
QMenuBar { background: #14161a; } QMenuBar::item:selected { background: #2a2e35; }
QMenu { background: #1d2025; border: 1px solid #2a2e35; } QMenu::item:selected { background: #2f343c; }
QListWidget#complist { background: transparent; border: none; }
QListWidget#complist::item { border-radius: 6px; }
QListWidget#complist::item:selected { background: #2a2418; border: 1px solid #f0a04b; }
QLineEdit { background: #101215; border: 1px solid #343944; border-radius: 4px; padding: 2px 6px;
            font-family: Menlo, monospace; }
QLabel#mono { font-family: Menlo, monospace; }
QPushButton#constraint { background: transparent; border: none; padding: 0 2px; text-align: left;
                         color: #8a9099; font-family: Menlo, monospace; }
QPushButton#constraint:hover { color: #f0a04b; }
QPushButton#constraint:disabled { color: #8a9099; }
QWidget#card QLabel, QWidget#compitem { background: transparent; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #5d636d; border-radius: 3px;
                       background: #101215; }
QCheckBox::indicator:checked { background: #f0a04b; border-color: #f0a04b; }
QCheckBox::indicator:disabled { border-color: #3a3f48; }
QCheckBox::indicator:checked:disabled { background: #8a6a40; border-color: #8a6a40; }
QCheckBox#vis::indicator { width: 6px; height: 12px; border-radius: 2px; }
QCheckBox#vis::indicator:checked { background: #c9cdd3; border-color: #c9cdd3; }
QPushButton#accent { background: #f0a04b; color: #14161a; border-color: #f0a04b; font-weight: bold; }
QPushButton#accent:disabled { background: #4a3a26; color: #8a7a66; border-color: #4a3a26; }
QCheckBox#inmodel::indicator { width: 6px; height: 12px; border-radius: 2px; }
QCheckBox#inmodel::indicator:checked { background: #f0a04b; border-color: #f0a04b; }
"""

MIN_CONTENT = (1200, 780)  # pixels: the window's content scrolls below this (small screens)
RENDER_DELAY_MS = 30  # coalesce drag events: render at most this often, with the latest state
SOURCE_PIXELS = 200  # source-plane display grid (per side)
MODES = ("Data", "Model", "Residual", "Data − model")  # the last in data units (e.g. quasar subtracted)
DISPLAYS = MODES + ("Source plane",)  # each has its own scaling (the Scale menu)
CMAPS = {"Residual": "CET-D1", "Source plane": "CET-L3"}  # the others: CET-L1
SCALE_LABELS = {"default": "Default (forwardgal)", "linear": "linear", "log": "log", "sqrt": "sqrt",
                "squared": "squared", "asinh": "asinh", "histeq": "histogram equalization"}
LIMIT_LABELS = {"default": "Default limits", "minmax": "min max", "zscale": "zscale", "99.5": "99.5 %",
                "99": "99 %", "98": "98 %", "95": "95 %", "90": "90 %", "user": "user (Scale Parameters…)"}


def theta_text(setup, tx, ty):
    """Status-bar text for image-plane position theta (arcsec): theta, beta (with a lens), RA/Dec."""
    ra, dec = setup.field.theta_to_sky(float(tx), float(ty))
    beta = ""
    if setup.lensed:
        bx, by = setup.field.backtrace(float(tx), float(ty))
        beta = f"beta = ({float(bx):+.4f}, {float(by):+.4f})\"   "
    return f"theta = ({tx:+.3f}, {ty:+.3f})\"   {beta}RA, Dec = {float(ra):.7f}, {float(dec):+.7f}"


def _short(name, n=26):
    """A long file name shortened in the middle: 'jw0667...f115w_i2d.fits'."""
    return name if len(name) <= n else name[: n // 2 - 1] + "…" + name[-(n // 2):]


def _contour_lines(mask):
    """Outline(s) of a boolean mask as (N, 2) arrays of (col, row) (0-based pixel centres)."""
    from contourpy import contour_generator

    return contour_generator(z=np.asarray(mask, float)).lines(0.5)


@dataclass
class Band:
    """One band tab (G5): its model file and, while it is not the active band, its unsaved state."""

    name: str
    path: Path  # the model file it shows (its own, or a history folder's model / bestfit)
    spec: dict | None = None  # its edited spec (None: as in ``path``)
    loaded_from: str | None = None
    clean_spec: dict | None = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("forwardgal")
        self.resize(1500, 860)
        self.setStyleSheet(STYLE)
        self.path = self.cfg = self.setup = self.live = self.spec = self.result = None
        self.rois = {}  # component name -> EllipseROI
        self.bend_handles = {}  # component name -> {m: TargetItem (triangle)} for bending modes
        self._dragging = None  # name of the ellipse being dragged (not re-placed while it moves)
        self._placing = False  # True while the program moves ellipses (their signals are ignored)
        self.scales = {d: Scale() for d in DISPLAYS}  # the Scale menu's settings, per display
        self.smooths = {d: Smooth() for d in DISPLAYS}  # the Smooth menu's settings, per display
        self._arrays = {}  # panel name -> (display, raw array, smoothed array) as last drawn (read-out)
        self._shown = {}  # panel name -> (display, (lo, hi)) as last drawn
        self._src_sample = None  # the first source-plane image (its default limits stay put while dragging)
        self._curves = {}  # (lens, cosmology, cutout) -> (critical curves, caustics)

        # Header: input files ("chips") and tool modes.
        bar = self.addToolBar("header")
        bar.setMovable(False)
        title = QLabel("<b>forwardgal</b> · model builder")
        bar.addWidget(title)
        self.chips = {}
        for key in ("data", "alpha", "psf"):
            lab = QLabel(f"{key}: -")
            lab.setObjectName("chip")
            bar.addWidget(lab)
            self.chips[key] = lab
        # z_src, editable as in the mockup (G6b), and the deflection rescaling it implies
        self.chips["z_src"] = QLabel("z_src")
        self.chips["z_src"].setObjectName("chip")
        bar.addWidget(self.chips["z_src"])
        self.z_src = QLineEdit("")
        self.z_src.setFixedWidth(64)
        self.z_src.setToolTip("source redshift: the deflection maps are rescaled to it (Enter applies)")
        self.z_src.editingFinished.connect(self._z_src_edited)
        bar.addWidget(self.z_src)
        lab = QLabel("")
        lab.setObjectName("chip")
        bar.addWidget(lab)
        self.chips["scale"] = lab
        # The tool modes on a row of their own, under the header (as in the mockup).
        self.addToolBarBreak()
        tools = self.addToolBar("tools")
        tools.setMovable(False)
        self.tools = QActionGroup(self)
        self.tools.setExclusive(True)
        for text, enabled in (("Select / move", True), ("Pan / zoom", True), ("Draw ellipse", True),
                              ("Edit mask", True)):
            act = QAction(text, self, checkable=True, enabled=enabled)
            self.tools.addAction(act)
            tools.addAction(act)
        self.tools.actions()[0].setChecked(True)
        self.tools.triggered.connect(self._tool_changed)

        # Panels.
        self.source = PlanePanel("Source plane", "drag the ellipses · β (arcsec)", "β_x [arcsec]  (east ←)",
                                 "β_y [arcsec]")
        self.image = PlanePanel("Image plane", "data + lensed model · θ (arcsec)", "θ_x [arcsec]  (east ←)",
                                "θ_y [arcsec]")
        # A second image-plane panel under the first (View menu; V3), e.g. Data above and Residual below,
        # with pan and zoom linked both ways. The draggable ellipses stay in the first.
        self.image2 = PlanePanel("Image plane (2)", "pan and zoom follow the panel above · θ (arcsec)",
                                 "θ_x [arcsec]  (east ←)", "θ_y [arcsec]")
        # (Not pyqtgraph's setXLink/setYLink: those line views up by screen position, and with the locked
        # aspect ratio two stacked panels would fight. Instead each copies the other's range.)
        self._linking = False
        self.image.view.sigRangeChanged.connect(lambda v, *_: self._follow(v, self.image2.view))
        self.image2.view.sigRangeChanged.connect(lambda v, *_: self._follow(v, self.image.view))
        self.mode, self.mode2 = QButtonGroup(self), QButtonGroup(self)
        for group, panel, start in ((self.mode, self.image, 0), (self.mode2, self.image2, 2)):
            for k, text in enumerate(MODES):
                b = QPushButton(text, checkable=True)
                group.addButton(b, k)
                panel.buttons.addWidget(b)
            group.button(start).setChecked(True)
            group.idClicked.connect(lambda _: self._show_image())
        self.source.drawn.connect(lambda *xy: self._drawn("source", *xy))
        # Edit mask (G6c): the mask polygon as an editable PolyLineROI in the image plane, applied on request
        self.mask_roi = None  # the polygon being edited (theta); kept while another tool is chosen
        self.apply_mask_button, self.revert_mask_button = QPushButton("Apply mask"), QPushButton("Revert mask")
        self.apply_mask_button.setToolTip("use the edited polygon as the mask (RA/Dec) and rebuild the render box")
        self.revert_mask_button.setToolTip("forget the edits: back to the current mask")
        self.apply_mask_button.clicked.connect(self.apply_mask)
        self.revert_mask_button.clicked.connect(self.revert_mask)
        for b in (self.apply_mask_button, self.revert_mask_button):
            b.hide()
            self.image.buttons.insertWidget(2, b)
        self.image.drawn.connect(lambda *xy: self._drawn("image", *xy))
        self.image2.drawn.connect(lambda *xy: self._drawn("image", *xy))
        # the colorbars (D4): contrast and bias of whatever display the panel shows
        for name, panel in (("source", self.source), ("image", self.image), ("image2", self.image2)):
            panel.stretch_changed.connect(lambda c, b, name=name: self._stretch(name, c, b))
        self.image2.hide()
        self.panel = ComponentsPanel()
        self.panel.selected.connect(self._highlight)
        self.panel.selection_changed.connect(self._highlight)
        self.panel.edited.connect(self._value_edited)
        self.panel.lock_changed.connect(self._lock_changed)
        self.panel.constraint_clicked.connect(self._edit_constraint)
        self.panel.tie_clicked.connect(self._edit_tie)
        self.panel.untie_clicked.connect(self._untie)
        self.panel.fit_fluxes_clicked.connect(self._fit_fluxes)
        self.panel.add_clicked.connect(self._add_clicked)
        self.panel.remove_clicked.connect(self._remove)
        self.panel.visibility_changed.connect(self._apply_visibility)
        self.panel.model_changed.connect(self._model_toggled)
        self.panel.point_options_clicked.connect(self.open_point_options)
        self.panel.mode_add.connect(lambda n, k, m: self._mode_edit(add_mode, n, k, m, "added"))
        self.panel.mode_remove.connect(lambda n, k, m: self._mode_edit(remove_mode, n, k, m, "removed"))
        self._markers = {}  # point component name -> its + markers (both planes, both image panels)
        self.dialog = None  # the open constraint dialog, if any
        self._colours = {}  # component name -> colour (ellipse and list dot)
        self.image_column = QSplitter(Qt.Vertical)
        self.image_column.addWidget(self.image)
        self.image_column.addWidget(self.image2)
        # Right column: components, then the fit card (G4).
        self.fitpanel = FitPanel()
        self.fitpanel.fit_clicked.connect(self.start_fit)
        self.fitpanel.cancel_clicked.connect(self.cancel_fit)
        self.fitpanel.nuts_clicked.connect(self.start_nuts)
        self.fitpanel.corner_clicked.connect(self.open_corner)
        self._corner = None  # the last NUTS run's corner.png
        self.process = None  # the running fit (QProcess) or None
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.panel, 1)
        rl.addWidget(self.fitpanel)
        split = QSplitter(Qt.Horizontal)
        for w in (self.source, self.image_column, right):
            split.addWidget(w)
        split.setSizes([540, 600, 360])
        # The model history (G3) under the planes; the divider between them can be dragged.
        self.history = HistoryPanel()
        self.history.save_clicked.connect(self.save_to_history)
        self.history.load_clicked.connect(self._history_load_clicked)
        self.history.trash_clicked.connect(self._history_trash)
        self.history.restore_clicked.connect(self._history_restore)
        self.history.use_in_band_clicked.connect(self.use_in_band_dialog)
        outer = QSplitter(Qt.Vertical)
        outer.addWidget(split)
        outer.addWidget(self.history)
        outer.setSizes([680, 200])
        # Band tabs (G5) above everything: one model file per band; only the active band is built.
        self.bands = []  # Band records (see _stash / switch_band)
        self.active = -1
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(self._tab_changed)
        add_band = QPushButton("+ band")
        add_band.setToolTip("open another band's model file in a new tab")
        add_band.clicked.connect(self.add_band_dialog)
        band_row = QWidget()
        br = QHBoxLayout(band_row)
        br.setContentsMargins(4, 2, 4, 0)
        br.addWidget(self.tabs)
        br.addWidget(add_band)
        br.addStretch(1)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(band_row)
        cl.addWidget(outer, 1)
        # Scrollable (a small screen): below this size the content scrolls instead of being squeezed or cut
        central.setMinimumSize(*MIN_CONTENT)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)  # larger windows: the content fills them, as before
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setWidget(central)
        self.setCentralWidget(self.scroll)
        self.history_dir = None  # the fits/ folder of the model file (history.history_dir_for)
        self.loaded_from = None  # the history folder now in the GUI (or None: a model file of its own)
        self._clean_spec = None  # the spec as last loaded or saved (unsaved changes: spec differs)
        self.question = None  # an open "load which values?" message box
        self.watcher = QFileSystemWatcher(self)  # new or trashed folders (e.g. a fit finishing) refresh the table
        self.watcher.directoryChanged.connect(lambda _: self.refresh_history())

        self.coords = QLabel("")
        self.fitinfo = QLabel("")
        self.statusBar().addWidget(self.coords, 1)
        self.statusBar().addPermanentWidget(self.fitinfo)
        self.image.moved.connect(lambda x, y: self._image_mouse(x, y, "image"))
        self.image2.moved.connect(lambda x, y: self._image_mouse(x, y, "image2"))
        self.source.moved.connect(self._source_mouse)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(RENDER_DELAY_MS)
        self.timer.timeout.connect(self.render)

        m = self.menuBar().addMenu("&File")
        for text, key, slot in [("&Open model...", QKeySequence.Open, self.open_dialog),
                                ("Save to &history", QKeySequence.Save, self.save_to_history),
                                ("&Save model as...", QKeySequence.SaveAs, self.save_dialog),
                                ("&Export data − model as FITS...", None, self.export_dialog),
                                ("&Quit", QKeySequence.Quit, self.close)]:
            a = QAction(text, self)
            if key is not None:
                a.setShortcut(key)
            a.setMenuRole(QAction.MenuRole.NoRole)
            a.triggered.connect(slot)
            m.addAction(a)

        s = self.menuBar().addMenu("&Settings")
        self.flux_units = QActionGroup(self)
        self.flux_units.setExclusive(True)
        for text, unit in (("Fluxes as AB magnitudes", "mag"), ("Fluxes as fluxes", "flux")):
            a = QAction(text, self, checkable=True)
            a.setData(unit)
            a.setMenuRole(QAction.MenuRole.NoRole)  # macOS: keep it in this menu (not the app menu)
            self.flux_units.addAction(a)
            s.addAction(a)
        self.flux_units.triggered.connect(lambda a: self.set_flux_unit(a.data()))
        s.addSeparator()
        a = QAction("Model settings…", self)
        a.setMenuRole(QAction.MenuRole.NoRole)  # (macOS would otherwise move "settings" to the app menu)
        a.triggered.connect(self.open_settings)
        s.addAction(a)

        v = self.menuBar().addMenu("&View")
        self.view_actions = {}
        for key, text, checkable, slot in (
                ("show", "Show all markers", False, lambda: self.panel.set_hidden(set())),
                ("hide", "Hide all markers", False, lambda: self.panel.set_hidden(set(self.panel.names))),
                (None, None, None, None),
                ("grid", "Grid", True, lambda on: self.set_grid(on)),
                ("curves", "Caustics and critical curves", True, lambda on: self.set_curves(on)),
                ("second", "Second image panel", True, lambda on: self._show_second(on))):
            if key is None:
                v.addSeparator()
                continue
            a = QAction(text, self, checkable=checkable)
            a.setMenuRole(QAction.MenuRole.NoRole)
            a.triggered.connect(slot)
            v.addAction(a)
            self.view_actions[key] = a
        self.view_actions["grid"].setChecked(True)

        sm = self.menuBar().addMenu("S&cale")
        self.scale_groups = {}  # display -> (scale QActionGroup, limits QActionGroup)
        for d in DISPLAYS:
            sub = sm.addMenu(d)
            groups = []
            for field, labels in (("scale", SCALE_LABELS), ("limits", LIMIT_LABELS)):
                g = QActionGroup(self)
                g.setExclusive(True)
                for key, text in labels.items():
                    a = QAction(text, self, checkable=True)
                    a.setData(key)
                    a.setMenuRole(QAction.MenuRole.NoRole)
                    if field == "limits" and key == "user":
                        a.setEnabled(False)  # set by Scale Parameters... (D5)
                    g.addAction(a)
                    sub.addAction(a)
                g.triggered.connect(lambda a, d=d, field=field: self.set_scale(d, **{field: a.data()}))
                groups.append(g)
                sub.addSeparator()
            params = QAction("Scale Parameters…", self)
            params.setMenuRole(QAction.MenuRole.NoRole)
            params.triggered.connect(lambda _=False, d=d: self.open_scale_parameters(d))
            sub.addAction(params)
            reset = QAction("Default scaling", self)
            reset.setMenuRole(QAction.MenuRole.NoRole)
            reset.triggered.connect(lambda _=False, d=d: self.reset_scale(d))
            sub.addAction(reset)
            self.scale_groups[d] = tuple(groups)
        sm.addSeparator()
        a = QAction("All to default scaling", self)
        a.setMenuRole(QAction.MenuRole.NoRole)
        a.triggered.connect(lambda: self.reset_scale(None))
        sm.addAction(a)
        self._show_scale_checks()

        mm = self.menuBar().addMenu("S&mooth")
        self.smooth_actions = {}  # display -> ("Smooth" QAction, function QActionGroup)
        for d in DISPLAYS:
            sub = mm.addMenu(d)
            on = QAction("Smooth", self, checkable=True)
            on.setMenuRole(QAction.MenuRole.NoRole)
            on.triggered.connect(lambda checked, d=d: self.set_smooth(d, on=checked))
            sub.addAction(on)
            sub.addSeparator()
            g = QActionGroup(self)
            g.setExclusive(True)
            for f in FUNCTIONS:
                a = QAction(f, self, checkable=True)
                a.setData(f)
                a.setMenuRole(QAction.MenuRole.NoRole)
                g.addAction(a)
                sub.addAction(a)
            g.triggered.connect(lambda a, d=d: self.set_smooth(d, function=a.data()))
            sub.addSeparator()
            params = QAction("Smooth Parameters…", self)
            params.setMenuRole(QAction.MenuRole.NoRole)
            params.triggered.connect(lambda _=False, d=d: self.open_smooth_parameters(d))
            sub.addAction(params)
            self.smooth_actions[d] = (on, g)
        self._show_smooth_checks()
        self.view_actions["curves"].setChecked(True)
        self._curve_items = []  # the drawn caustics and critical curves (View toggle)

    # --- model files ------------------------------------------------------------

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open forwardgal model", "", "Model files (*.yaml *.yml)")
        if path:
            self.load(path)

    def load(self, path):
        """Read and build a model YAML, draw both planes and render (a few seconds)."""
        path = Path(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cfg = read_yaml(path)
            setup = build(cfg, base_dir=path.parent)
            live = LiveModel(setup)
        except Exception as e:  # show any problem in the file to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could not load the model", f"{path}\n\n{type(e).__name__}: {e}")
            return False
        # setup.cfg: old quasar: section upgraded, RA/Dec points resolved to x0, y0 (what LiveModel needs)
        cfg = setup.cfg
        self.path, self.cfg, self.setup, self.live = path, cfg, setup, live
        self.spec = copy.deepcopy(cfg["model"])
        # Fluxes are parameters (F4a): a file without them gets them from ONE linear solve, now; after
        # this only the user (or "Fit fluxes") changes them, never a drag.
        self.spec, added = add_flux_parameters(self.spec, live.render().amplitudes, live.zeropoint, live.pixscale)
        if added:
            self.statusBar().showMessage(f"flux parameters added from one linear solve: {', '.join(added)}", 8000)
        self.setWindowTitle(f"forwardgal - {path.name}")
        lensed = setup.lensed
        self._show_chips()
        self._settings_changed = False
        self.source.setVisible(lensed)  # no lens: no source plane
        self.panel.add_button.setToolTip(
            "a new " + ("source" if lensed else "image") + "-plane Sérsic at the centre of that plane's view; "
            "its flux is fitted once to the light the others leave over")
        self._draw_planes()
        self.render()
        # The history this model belongs to; a file inside it (fits/NNNN_label/...) is that folder's model.
        self.history_dir = history_dir_for(path, cfg)
        self.fitpanel.starts.setValue(int(cfg.get("fit", {}).get("n_starts", 5)))
        in_history = FOLDER_RE.match(path.parent.name) and path.parent.parent == self.history_dir
        self.loaded_from = path.parent.name if in_history else None
        self._clean_spec = copy.deepcopy(self.spec)
        self.refresh_history()
        self._band_loaded(path)
        QApplication.restoreOverrideCursor()
        return True

    def save_dialog(self):
        if self.cfg is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save forwardgal model", str(self.path or ""),
                                              "Model files (*.yaml)")
        if path:
            self.save(path)

    def save(self, path):
        """Write the configuration with the current (edited) model spec."""
        self.cfg["model"] = restore_radec(self.spec, self.setup.radec)  # unmoved RA/Dec points: as given
        cfg = absolute_paths(self.cfg, self.path.parent) if Path(path).parent != self.path.parent else self.cfg
        write_yaml(cfg, path, header="forwardgal model file (saved from the GUI)")
        self.statusBar().showMessage(f"saved {path}", 5000)

    # --- model history (G3) -----------------------------------------------------------

    def unsaved(self):
        """True if the model (or a setting) was changed since it was loaded or saved."""
        return self._clean_spec is not None and (self.spec != self._clean_spec
                                                 or getattr(self, "_settings_changed", False))

    # --- settings (G6b) ----------------------------------------------------------------------

    def _show_chips(self):
        """Header: input files, z_src (editable) and the deflection rescaling it implies."""
        from forwardgal.cosmology import Cosmology, deflection_scale

        cfg, lensed = self.cfg, self.setup.lensed
        files = {"data": cfg["data"]["image"]["file"], "psf": cfg["psf"]["file"],
                 "alpha": cfg["lens"]["deflection_x"] if lensed else None}
        for key, label in (("data", "data"), ("alpha", "α"), ("psf", "PSF")):
            name = _short(Path(files[key]).name) if files[key] else "none"
            self.chips[key].setText(f"{label}: {name}")
            self.chips[key].setToolTip(str(files[key]) if files[key] else
                                       "no lens: image plane only, theta from the reference position")
        self.z_src.setEnabled(lensed)
        self.z_src.setText(f"{cfg['cosmology']['z_src']:g}" if lensed else "")
        self.chips["z_src"].setText("z_src" if lensed else "no lens")
        if lensed:
            c, z_ref = cfg["cosmology"], cfg["lens"]["z_ref"]
            f = deflection_scale(c["z_lens"], z_ref, c["z_src"], Cosmology(c["H0"], c["Om0"], c["Ode0"]))
            self.chips["scale"].setText(f"α × {f:.3f}")
            self.chips["scale"].setToolTip(f"deflections rescaled from z_ref = {z_ref:g} to z_src = {c['z_src']:g}: "
                                           "f = (D_ls/D_s)(z_src) / (D_ls/D_s)(z_ref)")
        self.chips["scale"].setVisible(lensed)

    def apply_settings(self, sections, model=None):
        """Replace configuration sections (cosmology, render, mask, fit, psf) and rebuild, keeping the model
        (or using ``model``, a spec as in a file: e.g. a point given by ra/dec, G6d).

        Returns None (done) or an error message (nothing changed). Counts as an unsaved change.
        """
        if self.live is None:
            return "no model loaded"
        cfg = copy.deepcopy(self.cfg)
        cfg.update(copy.deepcopy(sections))
        cfg["model"] = copy.deepcopy(model) if model is not None else restore_radec(self.spec, self.setup.radec)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            setup = build(cfg, base_dir=self.path.parent)
            live = LiveModel(setup)
        except Exception as e:  # e.g. a box outside the deflection maps: keep the old set-up
            return f"{type(e).__name__}: {e}"
        finally:
            QApplication.restoreOverrideCursor()
        self.cfg, self.setup, self.live = setup.cfg, setup, live
        self.spec = copy.deepcopy(setup.cfg["model"])
        self._settings_changed = True
        self._show_chips()
        self._draw_planes()
        self.render()
        self.statusBar().showMessage(f"settings applied ({', '.join(sorted(sections)) or 'model'}); "
                                     "Save to history to keep them", 8000)
        return None

    def apply_point_options(self, name, changes):
        """Point options… -> its position (RA/Dec), snap limit, PSF and reference; rebuild (G6d).

        changes: as PointDialog.changes(). The point's template is made by config.build, so the set-up
        is built again; every other component (and the point's flux) stays as it is.
        """
        spec = restore_radec(self.spec, self.setup.radec)  # unmoved RA/Dec points as given
        comp = next(c for c in spec["components"] if c["name"] == name)
        comp["params"] = {k: v for k, v in comp["params"].items() if k not in ("x0", "y0")}
        comp["ra"], comp["dec"] = float(changes["ra"]), float(changes["dec"])
        comp["max_snap"] = changes["max_snap"]
        sections = {}
        if changes["psf"] is None:
            comp.pop("psf", None)
            if changes.get("global_reference") != self.cfg["psf"].get("reference", "centroid"):
                sections["psf"] = dict(self.cfg["psf"], reference=changes["global_reference"])
        else:
            comp["psf"] = dict(changes["psf"])
        err = self.apply_settings(sections, model=spec)
        if err is None:
            pt = self.setup.points[name]
            self.statusBar().showMessage(f"{name}: theta = ({pt.theta[0]:+.4f}, {pt.theta[1]:+.4f})\", "
                                         f"snapped by {pt.snap:.3f} px, PSF {Path(pt.psf).name}", 8000)
        return err

    def open_point_options(self, name):
        """The card's "Point options…" (non-blocking)."""
        from forwardgal.gui.point_dialog import PointDialog

        if self.live is None or name not in self.setup.points:
            return
        comp = next(c for c in self.spec["components"] if c["name"] == name)
        self.dialog = PointDialog(name, comp, self.setup.points[name], self.cfg["psf"],
                                  lambda ch: self.apply_point_options(name, ch), self)
        self.dialog.open()

    def open_settings(self):
        """Settings -> Model settings… (non-blocking)."""
        from forwardgal.gui.settings_dialog import SettingsDialog

        if self.live is None:
            return
        self.dialog = SettingsDialog(self.cfg, self.setup.lensed, self.apply_settings, self)
        self.dialog.open()

    def _z_src_edited(self):
        if self.live is None or not self.setup.lensed:
            return
        try:
            z = parse_value(self.z_src.text())
        except ValueError as e:
            self.statusBar().showMessage(f"z_src: {e}", 6000)
            self._show_chips()
            return
        if z == self.cfg["cosmology"]["z_src"]:
            return
        if not z > self.cfg["cosmology"]["z_lens"]:
            self.statusBar().showMessage("z_src must be beyond the lens (z_src > z_lens)", 6000)
            self._show_chips()
            return
        err = self.apply_settings({"cosmology": dict(self.cfg["cosmology"], z_src=z)})
        if err:
            self.statusBar().showMessage(f"z_src: {err}", 8000)
            self._show_chips()

    def refresh_history(self):
        """Re-read the history folder into the table (and watch it for changes)."""
        if self.history_dir is None:
            return
        hd = Path(self.history_dir)
        if hd.is_dir() and str(hd) not in self.watcher.directories():
            self.watcher.addPath(str(hd))
        self.history.set_rows(list_history(hd), self.loaded_from, list_trash(hd),
                              title=self.cfg.get("label", "") if self.cfg else None)
        self._update_nuts()

    def save_to_history(self):
        """Save the current model as a new numbered draft (history.save_draft); it becomes the loaded one."""
        if self.live is None:
            return None
        cfg = absolute_paths(dict(self.cfg, model=restore_radec(self.spec, self.setup.radec)), self.path.parent)
        r = self.result
        folder = save_draft(self.history_dir, cfg, label=cfg.get("label"), started_from=self.loaded_from,
                            chi2=r.chi2, dof=r.dof, n_pixels=int(self.live.mask.sum()))
        self.loaded_from, self._clean_spec = folder.name, copy.deepcopy(self.spec)
        self._settings_changed = False
        self.refresh_history()
        self.statusBar().showMessage(f"saved {folder.name} to the history (not fitted)", 6000)
        return folder

    def load_history(self, folder, which="bestfit"):
        """Load history folder ``folder`` (its name): a fit's "bestfit" or "start" values, a draft's model."""
        p = Path(self.history_dir) / folder
        path = p / ("bestfit.yaml" if which == "bestfit" and (p / "bestfit.yaml").exists() else "model.yaml")
        ok = self.load(path)
        if ok:
            what = {"bestfit.yaml": "best-fit values", "model.yaml": "starting values"}[path.name]
            self.statusBar().showMessage(f"loaded {folder}" + (f" ({what})" if (p / "result.yaml").exists() else ""),
                                         6000)
        return ok

    def _history_load_clicked(self, folder):
        """Load: a fit asks which values; unsaved changes are mentioned first (non-blocking box)."""
        is_fit = (Path(self.history_dir) / folder / "result.yaml").exists()
        if not is_fit and not self.unsaved():
            self.load_history(folder)
            return
        box = QMessageBox(self)
        box.setWindowTitle(f"Load {folder}")
        text = "Load the fit's best-fit values, or the starting values it was run with?" if is_fit else \
            f"Load {folder}?"
        if self.unsaved():
            text += "\n\nThe current model has unsaved changes; they will be lost (Save to history first?)."
        box.setText(text)
        choices = {}
        if is_fit:
            choices[box.addButton("Best-fit values", QMessageBox.AcceptRole)] = "bestfit"
            choices[box.addButton("Starting values", QMessageBox.AcceptRole)] = "start"
        else:
            choices[box.addButton("Load", QMessageBox.AcceptRole)] = "bestfit"
        box.addButton(QMessageBox.Cancel)
        box.buttonClicked.connect(lambda b: b in choices and self.load_history(folder, choices[b]))
        self.question = box
        box.open()

    # --- fitting (G4): forwardgal.run in a separate process ----------------------------------

    def start_fit(self, selected=False):
        """Fit the current model: saved to the history (a draft) first, then forwardgal.run on it.

        selected : True: only the selected components ("Fit selected"; run --only).
        The fit runs in its own process (QProcess): the window stays responsive,
        and its output is shown in the fit card. When it is done, its best fit is
        loaded (unless the model was changed meanwhile).
        """
        if self.live is None or self.process is not None:
            return False
        if self.live.off:
            self.statusBar().showMessage(f"Fit: tick \"in model\" for {', '.join(sorted(self.live.off))} first "
                                         "(a fit always uses the whole model)", 8000)
            return False
        only = self.panel.selected_names() if selected else None
        if selected and not only:
            self.statusBar().showMessage("Fit selected: select one or more components first", 5000)
            return False
        hd = Path(self.history_dir)
        if self.loaded_from and not self.unsaved() and (hd / self.loaded_from / "draft.yaml").exists():
            draft = self.loaded_from  # this exact model is already a draft
        else:
            draft = self.save_to_history().name
        args = ["-u", "-m", "forwardgal.run", str(hd / draft / "model.yaml"),
                "--starts", str(self.fitpanel.starts.value()), "--every", "10"]
        if only:
            args += ["--only", ",".join(only)]
        self._start_process(args, "fit", f"fitting {draft}" + (f" (only {', '.join(only)})" if only else "") + " ...")
        return True

    def start_nuts(self, warmup=500, samples=1000, chains=2):
        """Run NUTS -> posteriors from the loaded fit (python -m forwardgal.posterior, own process)."""
        if self.live is None or self.process is not None or not self._nuts_allowed()[0]:
            return False
        folder = Path(self.history_dir) / self.loaded_from
        args = ["-u", "-m", "forwardgal.posterior", str(folder), "--warmup", str(warmup), "--samples", str(samples),
                "--chains", str(chains)]
        self._start_process(args, "nuts", f"NUTS from {self.loaded_from}: {chains} x ({warmup} + {samples}) ...")
        return True

    def _nuts_allowed(self):
        """(allowed, why not): NUTS starts at a fit, so an unchanged fit must be loaded."""
        if self.process is not None:
            return False, "a fit or NUTS run is going on"
        if not self.loaded_from or not (Path(self.history_dir) / self.loaded_from / "result.yaml").exists():
            return False, "load a fit first (NUTS starts at a fit's best point)"
        if self.unsaved():
            return False, "the loaded fit was changed: NUTS starts at the fit as saved (load it again)"
        return True, ""

    def _update_nuts(self):
        self.fitpanel.set_nuts_enabled(*self._nuts_allowed())

    def open_corner(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if self._corner is not None and self._corner.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._corner)))

    def _start_process(self, args, kind, text):
        """Run ``python args`` (a fit or NUTS) in its own process; its output goes to the fit card."""
        self._fit_spec, self._fit_log, self._kind = copy.deepcopy(self.spec), [], kind
        self._job_band, self._job_hd = self.active, Path(self.history_dir)  # the job belongs to this band
        self.process = p = QProcess(self)
        p.setProgram(sys.executable)
        p.setArguments(args)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._fit_output)
        p.finished.connect(self._fit_finished)
        self.fitpanel.set_running(True)
        self.fitpanel.set_progress(text)
        p.start()
        self._update_tabs()

    def cancel_fit(self):
        if self.process is not None:
            self._fit_cancelled = True
            self.process.kill()

    def _fit_output(self):
        import re

        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]  # (progress bars rewrite with \r)
        self._fit_log += lines
        if lines:
            self.fitpanel.set_progress(lines[-1][:160])

    def _fit_finished(self, code, _status=None):
        import re

        if self.process.bytesAvailable():  # the last lines
            self._fit_output()
        cancelled = getattr(self, "_fit_cancelled", False)
        self.process, self._fit_cancelled = None, False
        self.fitpanel.set_running(False)
        self._update_tabs()
        self.refresh_history()
        self._update_nuts()
        if cancelled:
            self.fitpanel.set_progress("cancelled (nothing saved)")
            return
        m = next((re.search(r"; saved (.+?)(?: \(\d+ s in total\))?$", ln) for ln in reversed(self._fit_log)
                  if "; saved " in ln), None)
        if code != 0 or m is None:
            tail = " | ".join(self._fit_log[-3:])
            self.fitpanel.set_progress(f"the fit failed (exit {code}): {tail}")
            self.statusBar().showMessage("the fit failed: see the fit card", 8000)
            return
        folder = Path(m.group(1)).name
        if self._kind == "nuts":
            self._corner = self._job_hd / folder / "corner.png"
            self.fitpanel.corner.setEnabled(self._corner.exists())
        if self._job_band != self.active:  # its band is not shown now: loaded when the user switches to it
            b = self.bands[self._job_band]
            b.path, b.spec, b.loaded_from, b.clean_spec = self._job_hd / folder / "bestfit.yaml", None, folder, None
            self._update_tabs()
            self.fitpanel.set_progress(f"done: {folder} in {b.name} (shown when you switch to that band)")
        elif self.spec == self._fit_spec:
            self.load_history(folder, "bestfit")
            self.fitpanel.set_progress(f"done: {folder} (" + ("posteriors in result.yaml; corner plot"
                                                              if self._kind == "nuts" else "best fit loaded") + ")")
        else:
            self.fitpanel.set_progress(f"done: {folder} (not loaded: the model was changed meanwhile)")

    # --- bands (G5) -----------------------------------------------------------------------

    def _band_loaded(self, path):
        """After a load: the active band now shows ``path`` (the first load makes the first band)."""
        name = band_name(self.cfg, path.parent)
        if self.active < 0:
            self.bands.append(Band(name, path))
            self.active = 0
            self.tabs.blockSignals(True)
            self.tabs.addTab(name)
            self.tabs.setCurrentIndex(0)
            self.tabs.blockSignals(False)
        b = self.bands[self.active]
        b.name, b.path, b.spec, b.loaded_from, b.clean_spec = name, path, None, None, None
        self._update_tabs()

    def _update_tabs(self):
        """Tab texts: band · the history folder it shows · a running job."""
        for i, b in enumerate(self.bands):
            here = self.loaded_from if i == self.active else b.loaded_from
            job = self.process is not None and getattr(self, "_job_band", None) == i
            self.tabs.setTabText(i, b.name + (f" · {here.split('_')[0]}" if here else "")
                                 + (f" · {self._kind}…" if job else ""))
            self.tabs.setTabToolTip(i, str(b.path))

    def _stash(self):
        """Keep the active band's state (its edits) while another band is shown."""
        if 0 <= self.active < len(self.bands) and self.live is not None:
            b = self.bands[self.active]
            b.path, b.spec, b.loaded_from = self.path, copy.deepcopy(self.spec), self.loaded_from
            b.clean_spec = copy.deepcopy(self._clean_spec)

    def switch_band(self, i):
        """Show band ``i``: built again from its file (a few seconds), its unsaved edits put back."""
        if i == self.active or not 0 <= i < len(self.bands):
            return
        self._stash()
        prev, self.active = self.active, i
        b = self.bands[i]
        kept = (b.spec, b.loaded_from, b.clean_spec)
        if not self.load(b.path):
            self.active = prev
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(prev)
            self.tabs.blockSignals(False)
            return
        if kept[0] is not None:  # its edits (and which history folder they came from)
            self.spec, self.loaded_from, self._clean_spec = kept
            self.render()
            self.refresh_history()
            b.spec, b.loaded_from, b.clean_spec = kept
        self._update_tabs()

    def _tab_changed(self, i):
        self.switch_band(i)

    def add_band(self, path):
        """Open the model file ``path`` as a new band tab (and show it)."""
        self._stash()
        prev = self.active
        self.bands.append(Band(Path(path).stem, Path(path)))
        self.active = len(self.bands) - 1
        self.tabs.blockSignals(True)
        self.tabs.addTab(Path(path).stem)
        self.tabs.setCurrentIndex(self.active)
        self.tabs.blockSignals(False)
        if self.load(path):
            return True
        self.bands.pop()  # could not be built: back to the previous band
        self.tabs.blockSignals(True)
        self.tabs.removeTab(len(self.bands))
        self.active = prev
        self.tabs.setCurrentIndex(prev)
        self.tabs.blockSignals(False)
        return False

    def add_band_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open another band's model", "", "Model files (*.yaml *.yml)")
        if path:
            self.add_band(path)

    def use_in_band(self, folder, target, ranges):
        """Seed band ``target`` (an index, or a model file Path for a new band) from this band's fit ``folder``.

        The seeded model (seed.seed_into: the target's own set-up, this fit's shapes within ``ranges``,
        fluxes solved afresh) is saved as a draft in the TARGET band's history, started_from
        "<this band>:<folder>", and shown in that band.
        """
        from forwardgal.seed import seed_into

        source = self.bands[self.active].name
        if isinstance(target, int):
            b = self.bands[target]
            tpath = b.path
        else:
            tpath = Path(target)
        tcfg = read_yaml(tpath)
        seeded = seed_into(Path(self.history_dir) / folder, tcfg, tpath.parent, ranges)
        draft = save_draft(history_dir_for(tpath, tcfg), seeded, label=seeded.get("label"),
                           started_from=f"{source}:{folder}")
        if isinstance(target, int):
            b.path, b.spec, b.loaded_from, b.clean_spec = draft / "model.yaml", None, draft.name, None
            self.tabs.setCurrentIndex(target)  # -> switch_band
        else:
            self.add_band(draft / "model.yaml")
        self.statusBar().showMessage(f"{source} {folder} -> {self.bands[self.active].name}: {draft.name} "
                                     "(fluxes from one linear solve; fit it next)", 10000)
        return draft

    def use_in_band_dialog(self, folder):
        """History row "Use in band...": choose the target band and the ranges (non-blocking)."""
        from forwardgal.gui.band_dialog import UseInBandDialog

        others = [i for i in range(len(self.bands)) if i != self.active]

        def apply(choice, ranges):
            if choice is None:
                path, _ = QFileDialog.getOpenFileName(self, "The target band's model file", "",
                                                      "Model files (*.yaml *.yml)")
                if not path:
                    return "no model file chosen"
                target = Path(path)
            else:
                target = others[choice]
            try:
                self.use_in_band(folder, target, ranges)
            except (ValueError, KeyError, OSError) as e:
                return f"{type(e).__name__}: {e}"
            return None

        self.dialog = UseInBandDialog(self.bands[self.active].name, folder, [self.bands[i].name for i in others],
                                      apply, parent=self)
        self.dialog.open()

    def closeEvent(self, event):  # noqa: N802 (Qt's name)
        if self.process is not None:  # do not leave a fit running without its window
            self.process.kill()
            self.process.waitForFinished(5000)
        super().closeEvent(event)

    def _history_trash(self, folder):
        try:
            trash(self.history_dir, folder)
        except OSError as e:
            self.statusBar().showMessage(f"trash {folder}: {e}", 8000)
            return
        if folder == self.loaded_from:  # the model stays in the GUI, but no longer as that folder
            self.loaded_from = None
        self.refresh_history()
        self.statusBar().showMessage(f"moved {folder} to the trash (Trash: restore it)", 6000)

    def _history_restore(self, name):
        try:
            restore(self.history_dir, name)
        except OSError as e:
            self.statusBar().showMessage(f"restore {name}: {e}", 8000)
            return
        self.refresh_history()
        self.statusBar().showMessage(f"restored {name}", 6000)

    # --- drawing ------------------------------------------------------------------

    def _draw_planes(self):
        s, live = self.setup, self.live
        for p in (self.source, self.image, self.image2):
            p.clear_overlays()
        self._markers = {}
        if self.mask_roi is not None:  # a new set-up: its own mask (shown again when Edit mask is chosen)
            self.image.plot.removeItem(self.mask_roi)
            self.mask_roi = None
        for roi in self.rois.values():
            roi.getViewBox().removeItem(roi)
        self.rois = {}
        for handles in self.bend_handles.values():
            for h in handles.values():
                h.getViewBox().removeItem(h)
        self.bend_handles = {}

        # Image plane: the render box placed in theta.
        A, t, _ = pixel_to_theta(live.grid)
        self._theta_to_pixel = (np.linalg.inv(A), t)  # (col, row) = A^-1 (theta - t): the hover read-out
        self._box_transform = image_qtransform(A, t)
        to_theta = lambda cr: cr @ A.T + t  # noqa: E731  (N, 2) (col, row) -> theta
        self._soft = float(np.median(live.sigma[live.mask]))
        self._data_sample = np.asarray(live.data)[live.mask]  # default limits of Data, Model, Data − model
        for line in _contour_lines(live.mask):
            for p in (self.image, self.image2):
                p.add_curve(to_theta(line), pg.mkPen("w", width=1.2, style=Qt.DashLine))
        corners = to_theta(np.array([[-0.5, -0.5], [live.data.shape[1] - 0.5, live.data.shape[0] - 0.5]]))
        # once per deflection field: keyed by what makes it (not id(): a rebuilt field may reuse the address)
        d = self.cfg["data"]
        key = repr((self.cfg.get("lens"), self.cfg.get("cosmology"), d.get("centre"), d.get("reference"),
                    d.get("size_arcsec")))
        if key not in self._curves:
            cc = critical_curves(s.field)
            self._curves[key] = (cc, caustics(s.field, cc))
        cc, ca = self._curves[key]
        self._curve_items = []
        for c in cc:
            for p in (self.image, self.image2):
                self._curve_items.append(p.add_curve(c, pg.mkPen(TEAL, width=1, style=Qt.DotLine)))
        for name, pt in s.points.items():  # point components (the quasar): image position and traced beta
            self._markers[name] = [self.image.add_marker(*pt.theta, GOLD), self.image2.add_marker(*pt.theta, GOLD),
                                   self.source.add_marker(*pt.beta, GOLD)]
        self.image.show_range(corners[:, 0], corners[:, 1])

        # Source plane: a grid around the source-plane components.
        src = [v for n, v in live.values.items()
               if live.components[n].kind == "sersic" and live.components[n].plane == "source"]
        if src:
            cx = float(np.mean([v["x0"] for v in src]))
            cy = float(np.mean([v["y0"] for v in src]))
            half = max(1.0, 4.0 * max(v["R_e"] for v in src))
        else:
            cx, cy, half = 0.0, 0.0, 5.0
        n = SOURCE_PIXELS
        step = 2 * half / n
        g = (cx - half) + (np.arange(n) + 0.5) * step
        gy = (cy - half) + (np.arange(n) + 0.5) * step
        self._src_grid = np.meshgrid(g, gy)  # row-major: [row = y, col = x], x increasing with col
        self._src_transform = QTransform(step, 0, 0, step, cx - half, cy - half)
        self._src_geom = (cx - half, cy - half, step)  # grid corner and pixel size (read-out)
        self._src_sample = None
        for c in ca:
            self._curve_items.append(self.source.add_curve(c, pg.mkPen(TEAL, width=1.2)))
        self.set_curves(self.view_actions["curves"].isChecked())
        self.source.show_range((cx - half, cx + half), (cy - half, cy + half))

        self._colours = {}
        self._sync_rois()

    def _sync_rois(self):
        """Match the ellipses and colours to the components (after loading, adding or removing one).

        An ellipse and its list dot share a colour: points gold, the sky grey,
        Sersics the first colour not yet in use (so a component keeps its colour
        when another one is added or removed). The views are not moved.
        """
        live = self.live
        self._colours = {n: col for n, col in self._colours.items() if n in live.components}
        for name, c in live.components.items():
            if name in self._colours:
                continue
            if c.kind in ("point", "sky"):
                self._colours[name] = GOLD if c.kind == "point" else "#6b7280"
            else:
                used = set(self._colours.values())
                free = [col for col in COMPONENT_COLOURS if col not in used]
                self._colours[name] = free[0] if free else COMPONENT_COLOURS[len(self._colours) % 6]
        for name in [n for n in self.rois if n not in live.components]:  # removed components
            roi = self.rois.pop(name)
            roi.getViewBox().removeItem(roi)
        for name, c in live.components.items():  # one ellipse per Sersic, in its own plane
            if c.kind == "sersic" and name not in self.rois:
                self.rois[name] = self._make_roi(name, c)
        self._tool_changed()
        self._apply_visibility()

    # --- bending handles (triangles) --------------------------------------------------------

    def _sync_handles(self):
        """One draggable triangle per bending mode of each Sersic (in its plane); removed with the mode."""
        if self.live is None:
            return
        want = {(n, int(k[1:])) for n, c in self.live.components.items() if c.kind == "sersic"
                for k in c.params if k[0] == "b" and k[1:].isdigit()}
        for name in list(self.bend_handles):
            for m in list(self.bend_handles[name]):
                if (name, m) not in want:
                    h = self.bend_handles[name].pop(m)
                    h.getViewBox().removeItem(h)
            if not self.bend_handles[name]:
                del self.bend_handles[name]
        for name, m in sorted(want):
            if m in self.bend_handles.get(name, {}):
                continue
            c = self.live.components[name]
            h = pg.TargetItem(pos=(0.0, 0.0), size=13, symbol="t", movable=True,
                              pen=pg.mkPen(self._colours.get(name, "w"), width=1.5),
                              brush=pg.mkBrush(self._colours.get(name, "w")), label=f"b{m}",
                              labelOpts={"color": self._colours.get(name, "w"), "offset": (8, -8)})
            h.setToolTip(f"{name}: bending mode m = {m}; drag sideways to bend (the ridge of this mode alone "
                         "passes through the triangle)")
            h.setZValue(25)
            h.sigPositionChanged.connect(lambda item, name=name, m=m: self._handle_moved(name, m, item))
            (self.source if c.plane == "source" else self.image).plot.addItem(h)
            self.bend_handles.setdefault(name, {})[m] = h
        self._place_handles()
        self._apply_visibility()
        self._tool_changed()

    def _place_handles(self, skip=None):
        """Put the triangles where the current values say (all but ``skip`` = (name, m), being dragged)."""
        was, self._placing = self._placing, True
        try:
            for name, handles in self.bend_handles.items():
                v = self.live.values[name]
                pa = float(sersic_qpa(v)[1])
                ref = self.live.components[name].bend_pa_ref
                for m, h in handles.items():
                    if (name, m) != skip:
                        h.setPos(bend_handle_pos(v["x0"], v["y0"], v["R_e"], pa, ref, m, v[f"b{m}"]))
        finally:
            self._placing = was

    def _handle_moved(self, name, m, item):
        """A triangle dragged: b_m from its sideways offset; it is kept on its line (fixed distance along u)."""
        if self._placing:
            return
        v = self.live.values[name]
        pa = float(sersic_qpa(v)[1])
        ref = self.live.components[name].bend_pa_ref
        p = item.pos()
        b = bend_from_handle(v["x0"], v["y0"], v["R_e"], pa, ref, m, p.x(), p.y())
        self._placing = True
        try:
            item.setPos(bend_handle_pos(v["x0"], v["y0"], v["R_e"], pa, ref, m, b))  # back onto its line
        finally:
            self._placing = False
        try:
            spec = set_param(self.spec, name, f"b{m}", value=b)
        except (ValueError, KeyError) as e:  # locked, tied, or outside a bound
            self.statusBar().showMessage(f"{name}.b{m}: {e}", 5000)
            self._place_handles()
            return
        if name not in self.panel.selected_names():
            self._select(name)
        self.spec = spec
        self.timer.start()  # render after the drag settles (as for the ellipses)

    def _apply_visibility(self):
        """Show or hide each component's ellipse and + markers (the list's "show marker" boxes)."""
        hidden = self.panel.hidden
        for name, roi in self.rois.items():
            roi.setVisible(name not in hidden)
        for name, handles in self.bend_handles.items():
            for h in handles.values():
                h.setVisible(name not in hidden)
        for name, items in self._markers.items():  # (a removed point's markers are hidden too)
            shown = name not in hidden and self.live is not None and name in self.live.components
            for item in items:
                item.setVisible(shown)

    def _follow(self, src, dst):
        """Pan / zoom ``dst`` (the other image panel) to ``src``'s range; its locked aspect may widen one axis."""
        if self._linking:  # dst's own range change must not send it back
            return
        self._linking = True
        try:
            (x0, x1), (y0, y1) = src.viewRange()
            dst.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        finally:
            self._linking = False

    def _show_second(self, on):
        """View -> Second image panel: show it (at the first panel's range) or hide it."""
        self.image2.setVisible(on)
        if on:
            self._follow(self.image.view, self.image2.view)

    def set_curves(self, on):
        """View -> Caustics and critical curves: show or hide them in every plot."""
        for item in self._curve_items:
            item.setVisible(on)
        self.view_actions["curves"].setChecked(on)

    def set_grid(self, on):
        """View -> Grid: the coordinate grid of every plot on or off."""
        for p in (self.source, self.image, self.image2):
            p.plot.showGrid(x=on, y=on, alpha=0.15)
        self.view_actions["grid"].setChecked(on)

    def _make_roi(self, name, c):
        v = self.live.values[name]
        q, pa = (float(x) for x in sersic_qpa(v))
        pos, size, angle = ellipse_to_roi(v["x0"], v["y0"], v["R_e"], q, pa)
        roi = pg.EllipseROI(pos, size, angle=angle, pen=pg.mkPen(self._colours[name], width=2))
        roi.sigRegionChanged.connect(lambda r, name=name: self._roi_moved(name, r))
        roi.sigRegionChangeStarted.connect(lambda r, name=name: self._drag_started(name))
        roi.sigRegionChangeFinished.connect(lambda r: setattr(self, "_dragging", None))
        roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)  # a click selects it (sigClicked)
        # Ctrl-click (⌘-click on a Mac: Qt's ControlModifier is ⌘ there) adds it to the selection
        roi.sigClicked.connect(lambda r, ev, name=name: self._select(
            name, add=bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)))
        (self.source if c.plane == "source" else self.image).plot.addItem(roi)
        return roi

    def render(self):
        """Render the current spec (LiveModel) and refresh both planes and chi^2."""
        if self.live is None:
            return
        self.live.set_spec(self.spec)
        self.result = r = self.live.render()
        self._show_image()
        if self.setup.lensed:  # (no lens: no source plane to show)
            self._src_image = self.live.source_image(*self._src_grid, r.amplitudes)
            if self._src_sample is None and np.ptp(self._src_image) > 0:
                self._src_sample = np.array(self._src_image)  # (default limits fixed while dragging)
            self._show_source()
        off = sorted(self.live.off)
        self.fitinfo.setText(f"χ²/ν = {r.chi2 / r.dof:.4f}   (χ² = {r.chi2:.1f}, ν = {r.dof})"
                             + (f"   without {', '.join(off)}" if off else ""))
        self.panel.set_model(self.spec, self.live.values, r.amplitudes, self._colours, self._photometry_notes(r))
        self._show_flux_unit()
        self._update_nuts()  # (an edit makes the loaded fit "changed": NUTS off)
        self._sync_handles()
        self._place_rois(skip=self._dragging)  # e.g. a tied component follows, or a value typed in the card
        self._highlight(self.panel.current)

    def _place_rois(self, skip=None):
        """Put every ellipse where the current values say (all but ``skip``, the one being dragged)."""
        self._placing = True
        try:
            for name, roi in self.rois.items():
                if name == skip:
                    continue
                v = self.live.values[name]
                q, pa = (float(x) for x in sersic_qpa(v))
                pos, size, angle = ellipse_to_roi(v["x0"], v["y0"], v["R_e"], q, pa)
                st = roi.getState()
                now = (st["pos"].x(), st["pos"].y(), st["size"].x(), st["size"].y(), st["angle"])
                if np.allclose(now, (*pos, *size, angle), rtol=0, atol=1e-10):
                    continue
                roi.setAngle(angle, update=False)
                roi.setSize(size, update=False)
                roi.setPos(pos)  # one redraw, after all three
            self._place_handles()
        finally:
            self._placing = False

    def _show_image(self):
        if self.result is None:
            return
        for name, panel, group in (("image", self.image, self.mode), ("image2", self.image2, self.mode2)):
            mode = MODES[group.checkedId()]
            raw, _ = self._display_data(mode)
            arr = smooth(raw, self.smooths[mode])  # (display only; normalized over the pixels with values)
            self._arrays[name] = (mode, raw, arr)
            sample = np.asarray(arr)[self.live.mask]
            default, dsample = self._default_for(mode)
            y, lim = apply(arr, self.scales[mode], default, sample=sample, default_sample=dsample)
            cmap = pg.colormap.get(CMAPS.get(mode, "CET-L1"))
            panel.set_image(y, self._box_transform, (0.0, 1.0), cmap)
            panel.colorbar.set_state(cmap, self.scales[mode].contrast, self.scales[mode].bias)
            self._shown[name] = (mode, lim)

    def _display_data(self, display):
        """(the array shown, its pixels used for limits) of a display."""
        if display == "Source plane":
            src = self._src_image
            return src, np.asarray(src).ravel()
        arr = {"Data": self.live.data, "Model": self.result.model, "Residual": self.result.residual,
               "Data − model": self.live.data - self.result.model}[display]
        return arr, np.asarray(arr)[self.live.mask]

    def _default_for(self, display):
        """(Default, the pixels for its limits) of a display: forwardgal's look."""
        if display == "Residual":  # in sigma: linear -5 .. 5
            return Default(limits=("fixed", -5.0, 5.0)), None
        if display == "Source plane":  # asinh(v / noise), 0 .. 99.9 %, from the first source image (steady)
            return (Default(pre="asinh", soft=self._soft, limits=("zero_pct", 99.9)),
                    self._src_sample if self._src_sample is not None else self._src_image)
        # asinh(v / noise), limits from the data (so Model and Data − model compare with it)
        return Default(pre="asinh", soft=self._soft, limits=("pct", 0.5, 99.7)), self._data_sample

    def open_scale_parameters(self, display):
        """Scale -> display -> Scale Parameters...: the histogram with draggable limits (non-blocking)."""
        from forwardgal.gui.scale_dialog import ScaleDialog

        if self.result is None or (display == "Source plane" and getattr(self, "_src_image", None) is None):
            return
        arr, sample = self._display_data(display)
        default, dsample = self._default_for(display)
        sc = self.scales[display]
        _, (lo, hi) = apply(arr, sc, default, sample=sample, default_sample=dsample)
        if sc.scale == "default" and default.pre == "asinh":  # limits of asinh(v / s): back to data units
            lo, hi = (float(default.soft * np.sinh(x)) for x in (lo, hi))

        def apply_limits(lo, hi):
            if not lo < hi:
                return "the low limit must be below the high one"
            self.set_scale(display, limits="user", lo=lo, hi=hi)
            return None

        self.dialog = ScaleDialog(display, sample, lo, hi, apply_limits, self)
        self.dialog.open()

    def _show_source(self):
        if not self.setup.lensed or getattr(self, "_src_image", None) is None:
            return
        raw = self._src_image
        src = smooth(raw, self.smooths["Source plane"])
        self._arrays["source"] = ("Source plane", raw, src)
        default, dsample = self._default_for("Source plane")
        y, lim = apply(src, self.scales["Source plane"], default, default_sample=dsample)
        cmap = pg.colormap.get(CMAPS["Source plane"])
        self.source.set_image(y, self._src_transform, (0.0, 1.0), cmap)
        sc = self.scales["Source plane"]
        self.source.colorbar.set_state(cmap, sc.contrast, sc.bias)
        self._shown["source"] = ("Source plane", lim)

    # --- the Scale menu (D3) -------------------------------------------------------------------

    def set_scale(self, display, **changes):
        """Change one display's scaling (scale, limits, lo, hi, contrast, bias) and redraw."""
        try:
            self.scales[display] = self.scales[display].with_(**changes)
        except ValueError as e:
            self.statusBar().showMessage(f"Scale {display}: {e}", 6000)
        self._show_scale_checks()
        self._redraw()

    def _stretch(self, panel_name, contrast, bias):
        """A colorbar dragged: contrast and bias of the display that panel shows."""
        if panel_name in self._shown:
            self.set_scale(self._shown[panel_name][0], contrast=contrast, bias=bias)

    def reset_scale(self, display=None):
        """Default scaling for one display (None: all of them)."""
        for d in (DISPLAYS if display is None else (display,)):
            self.scales[d] = Scale()
        self._show_scale_checks()
        self._redraw()

    def _redraw(self):
        if self.result is not None:
            self._show_image()
            self._show_source()

    # --- the Smooth menu ---------------------------------------------------------------------

    def set_smooth(self, display, **changes):
        """Change one display's smoothing (on, function, radius, sigma) and redraw."""
        try:
            self.smooths[display] = self.smooths[display].with_(**changes)
        except ValueError as e:
            self.statusBar().showMessage(f"Smooth {display}: {e}", 6000)
        self._show_smooth_checks()
        self._redraw()

    def open_smooth_parameters(self, display):
        """Smooth -> display -> Smooth Parameters... (non-blocking); Apply turns smoothing on."""
        from forwardgal.gui.smooth_dialog import SmoothDialog

        def apply_smooth(s):
            self.smooths[display] = s
            self._show_smooth_checks()
            self._redraw()
            return None

        self.dialog = SmoothDialog(display, self.smooths[display], apply_smooth, self)
        self.dialog.open()

    def _show_smooth_checks(self):
        for d, (on, g) in self.smooth_actions.items():
            on.setChecked(self.smooths[d].on)
            for a in g.actions():
                a.setChecked(a.data() == self.smooths[d].function)

    def _show_scale_checks(self):
        for d, (gs, gl) in self.scale_groups.items():
            for g, key in ((gs, self.scales[d].scale), (gl, self.scales[d].limits)):
                for a in g.actions():
                    a.setChecked(a.data() == key)

    # --- interaction ----------------------------------------------------------------

    def _roi_moved(self, name, roi):
        if self._placing:  # moved by _place_rois, not by the user
            return
        self._dragging = name
        st = roi.getState()
        try:
            x0, y0, R_e, q, pa = roi_to_ellipse((st["pos"].x(), st["pos"].y()), (st["size"].x(), st["size"].y()),
                                                st["angle"])
        except NotImplementedError:
            self.statusBar().showMessage("Dragging needs roi_to_ellipse: TODO(Erik) in gui/geometry.py", 4000)
            return
        self.spec = set_ellipse(self.spec, name, x0=x0, y0=y0, R_e=R_e, q=q, pa_deg=pa)
        self.timer.start()  # (re)start: renders RENDER_DELAY_MS after the last change

    def _select(self, name, add=False):
        """The user clicked an ellipse: select it (add=True: toggle it in the selection), show its card."""
        self.panel.select(name, add=add)
        self._highlight()

    def _drag_started(self, name):
        """Dragging an ellipse selects it, unless it is already part of the selection."""
        if name not in self.panel.selected_names():
            self._select(name)

    def _highlight(self, *_):
        """Draw the selected components' ellipses thicker; the one shown in the card on top."""
        chosen = set(self.panel.selected_names())
        off = self.live.off if self.live is not None else set()
        for n, roi in self.rois.items():  # an "off" component: dashed
            roi.setPen(pg.mkPen(self._colours[n], width=3.5 if n in chosen else 2,
                                style=Qt.DashLine if n in off else Qt.SolidLine))
            roi.setZValue(20 if n == self.panel.current else 15 if n in chosen else 10)

    def _value_edited(self, name, key, text):
        """A value typed in the card: a checked spec edit, or a message and the old value back."""
        try:
            x = parse_value(text)
            comp = next(c for c in self.spec["components"] if c["name"] == name)
            if key in ("q", "pa_deg") and "e1" in comp["params"]:  # derived from (e1, e2)
                spec = set_ellipse(self.spec, name, **{key: x})
                check(spec)
            elif key[:3] == "phi" and f"c{key[3:]}" in comp["params"]:  # a Fourier mode's phase (G6a)
                spec = set_mode(self.spec, name, int(key[3:]), phi_deg=x)
            elif key[0] == "a" and key[1:].isdigit() and f"c{key[1:]}" in comp["params"]:  # its amplitude
                spec = set_mode(self.spec, name, int(key[1:]), a=x)
            else:
                spec = set_param(self.spec, name, key, value=x)
        except NotImplementedError as e:
            self.statusBar().showMessage(f"{e} (in forwardgal/edit.py)", 5000)
            self.panel.revert()
            return
        except (ValueError, KeyError) as e:  # SpecError is a ValueError
            self.statusBar().showMessage(f"{name}.{key}: {e}", 8000)
            self.panel.revert()
            return
        self.spec = spec
        self.render()

    def _as_qpa(self, spec, name, key):
        """(spec, converted): an (e1, e2) Sersic turned into (q, PA) if ``key`` is its derived q or PA (G2e)."""
        comp = next(c for c in spec["components"] if c["name"] == name)
        if key in ("q", "pa_deg") and "e1" in comp["params"]:
            return to_qpa(spec, name), True
        return spec, False

    def _mode_edit(self, edit, name, kind, m, verb):
        """ "+ mode" / "− mode": a checked spec edit (edit.add_mode / remove_mode)."""
        try:
            spec = edit(self.spec, name, kind, m)
        except (ValueError, KeyError) as e:  # SpecError is a ValueError
            self.statusBar().showMessage(f"{name}: {e}", 8000)
            return
        self.spec = spec
        self.render()
        self.statusBar().showMessage(f"{name}: {kind} mode m = {m} {verb}"
                                     + (" (at zero amplitude: the shape is unchanged)" if verb == "added" else ""),
                                     6000)

    def _lock_changed(self, name, key, locked):
        """A lock box clicked: fix the parameter at its value (locked) or free it again.

        q or PA of an (e1, e2) Sersic: the component is first turned into (q, PA) parameters at its current
        shape (edit.to_qpa), so that one of them can be fixed alone.
        """
        try:
            spec, converted = self._as_qpa(self.spec, name, key)
            spec = set_param(spec, name, key, free=not locked)
        except NotImplementedError:
            self.statusBar().showMessage("Locking q or PA needs to_qpa: TODO(Erik) in forwardgal/edit.py", 5000)
            self.panel.revert()
            return
        except (ValueError, KeyError) as e:  # SpecError is a ValueError
            self.statusBar().showMessage(f"{name}.{key}: {e}", 8000)
            self.panel.revert()
            return
        self.spec = spec
        self.render()
        v = self.live.values[name][key]
        self.statusBar().showMessage((f"{name} is now in (q, PA); " if converted else "")
                                     + f"{name}.{key} " + (f"locked at {v:g}" if locked else "free"), 6000)

    def _edit_constraint(self, name, key):
        """Open the bounds / prior dialog of name.key (non-blocking: ``open``, not ``exec``).

        q or PA of an (e1, e2) Sersic: converted to (q, PA) first (G2e), so it has bounds of its own.
        """
        try:
            spec, converted = self._as_qpa(self.spec, name, key)
        except NotImplementedError:
            self.statusBar().showMessage("Constraints on q or PA need to_qpa: TODO(Erik) in forwardgal/edit.py",
                                         5000)
            return
        if converted:
            self.spec = spec
            self.render()
            self.statusBar().showMessage(f"{name} is now in (q, PA) parameters (same shape)", 6000)
        p = next(c for c in self.spec["components"] if c["name"] == name)["params"][key]

        def apply(fields):
            try:
                spec = set_param(self.spec, name, key, **fields)
            except (ValueError, KeyError) as e:  # SpecError is a ValueError
                return str(e)
            self.spec = spec
            self.render()
            self.statusBar().showMessage(f"{name}.{key}: constraint set", 5000)
            return None

        self.dialog = ConstraintDialog(f"{name}.{key}", p, self.panel._cells[key][0].text(), apply, self)
        self.dialog.open()

    def _edit_tie(self, name):
        """Open the tie dialog for component ``name`` (non-blocking)."""
        rows = param_rows(self.spec, name, self.live.values[name])
        params = next(c for c in self.spec["components"] if c["name"] == name)["params"]
        # not tied, joint, point position, flux; and real parameters (not an (e1, e2) Sersic's derived q, PA)
        targets = [(r.key, r.label) for r in rows if r.lockable and r.key in params]
        try:
            tie_candidates(self.spec, name, targets[0][0]) if targets else None
        except NotImplementedError:
            self.statusBar().showMessage("Ties need tie_candidates: TODO(Erik) in forwardgal/edit.py", 5000)
            return

        def offset(key, source):
            comp, _, skey = source.partition(".")
            return self.live.values[name][key] - self.live.values[comp][skey]

        def apply(key, source, off):
            try:
                spec = set_tie(self.spec, f"{name}.{key}", source, off)
            except (ValueError, KeyError) as e:  # SpecError is a ValueError
                return str(e)
            self.spec = spec
            self.render()
            self.statusBar().showMessage(f"{name}.{key} tied to {source}", 5000)
            return None

        self.dialog = TieDialog(name, targets, lambda key: tie_candidates(self.spec, name, key), offset, apply,
                                self)
        self.dialog.open()

    def _untie(self, target):
        """Remove the tie of ``target``: it keeps its current value and becomes free to move."""
        comp, _, key = target.partition(".")
        value = self.live.values[comp][key]  # the tied value it has now
        try:
            spec = set_tie(self.spec, target, None)
            spec = set_param(spec, comp, key, value=value)  # stays where it is (its own value was ignored)
        except (ValueError, KeyError) as e:
            self.statusBar().showMessage(f"{target}: {e}", 8000)
            return
        self.spec = spec
        self.render()
        self.statusBar().showMessage(f"{target} untied", 5000)

    # --- adding and removing components (A3) -----------------------------------------

    def flux_unit(self):
        """'mag' if any non-sky component's flux is a magnitude (the Settings choice), else 'flux'."""
        return "mag" if any("mag" in c["params"] for c in self.spec["components"] if c["kind"] != "sky") \
            else "flux"

    def add_sersic(self, plane, x0, y0, R_e, q=0.8, pa_deg=0.0):
        """Add a Sersic (edit.add_sersic) with its flux fitted ONCE to the leftover light; select it.

        The leftover fit: LiveModel.render solves the amplitude of a component
        without a flux parameter by NNLS against the data minus all given
        fluxes, i.e. F = max(0, sum T (d - m) / s^2 / sum T^2 / s^2); that F
        becomes its flux parameter (photometry.add_flux_parameters), in the
        Settings unit. Returns the new name, or None (refused; message shown).
        """
        if self.live is None:
            return None
        name = new_name(self.spec)
        try:
            spec = add_sersic(self.spec, name, plane, x0, y0, R_e, q, pa_deg)
            self.live.set_spec(spec)
            amps = self.live.render().amplitudes
            unit = self.flux_unit() if self.live.zeropoint is not None else "flux"
            spec, _ = add_flux_parameters(spec, amps, self.live.zeropoint, self.live.pixscale, unit=unit)
        except (ValueError, KeyError) as e:  # SpecError is a ValueError
            self.live.set_spec(self.spec)
            self.statusBar().showMessage(f"Add Sérsic: {e}", 8000)
            return None
        self.spec = spec
        self.live.set_spec(spec)
        self._sync_rois()
        self.render()
        self._select(name)
        f = amps[name]
        self.statusBar().showMessage(f"added {name} ({plane} plane); flux fitted to the leftover light: "
                                     f"{f:.4g}" + ("  (nothing left over: started faint)" if f <= 0 else ""), 8000)
        return name

    def _add_clicked(self):
        """ "+ Add Sérsic": a small Sersic at the centre of the source-plane view (no lens: the image plane)."""
        if self.live is None:
            return
        plane, panel = ("source", self.source) if self.setup.lensed else ("image", self.image)
        (x_lo, x_hi), (y_lo, y_hi) = panel.view.viewRange()
        R_e = float(np.clip((x_hi - x_lo) / 20, 0.02, 1.0))
        self.add_sersic(plane, (x_lo + x_hi) / 2, (y_lo + y_hi) / 2, R_e)

    def _drawn(self, plane, xa, ya, xb, yb):
        """Draw ellipse tool: a drag along the major axis in ``plane`` adds a Sersic there (q starts at 0.6)."""
        x0, y0, R_e, q, pa = drag_to_ellipse((xa, ya), (xb, yb))
        if R_e < 0.01:  # a click, not a drag (R_e must be >= 0.01")
            self.statusBar().showMessage("Draw ellipse: drag from one end of the major axis to the other", 5000)
            return None
        return self.add_sersic(plane, x0, y0, R_e, q, pa)

    def _remove(self, name):
        """ "Remove": drop the component and any ties to or from it (edit.remove_component).

        A parameter that was tied to it keeps the value it had (as with "untie"): nothing else moves.
        """
        if self.live is None or name is None:
            return
        try:
            spec = remove_component(self.spec, name)
            dropped = sorted(t for t in self.spec.get("ties", {}) if t not in spec.get("ties", {}))
            for t in dropped:  # a parameter that followed the removed one keeps its value (as "untie")
                comp, _, key = t.partition(".")
                if comp != name:
                    spec = set_param(spec, comp, key, value=self.live.values[comp][key])
        except (ValueError, KeyError) as e:  # SpecError is a ValueError
            self.statusBar().showMessage(f"Remove {name}: {e}", 8000)
            return
        self.spec = spec
        self.live.set_spec(spec)
        self._sync_rois()
        self.render()
        self.statusBar().showMessage(f"removed {name}" + (f" and the ties {', '.join(dropped)}" if dropped else ""),
                                     8000)

    def _model_toggled(self):
        """An "in model" box: render without the unticked components (they stay in the spec)."""
        if self.live is None:
            return
        self.live.off = set(self.panel.off)
        self.render()

    def export_fits(self, path, area="box"):
        """Write data − model (the components now in the model) to a FITS file; area 'box' or 'cutout'."""
        from forwardgal.export import data_minus_model_hdu

        subtracted = [n for n in self.live.components if n not in self.live.off]
        hdu = data_minus_model_hdu(self.setup.img, self.setup.region, self.result.model, subtracted, area=area)
        hdu.writeto(path, overwrite=True)
        self.statusBar().showMessage(f"wrote {path}: data − ({', '.join(subtracted) or 'nothing'})", 8000)
        return subtracted

    def export_dialog(self):
        """File -> Export data − model: choose the area, then the file."""
        if self.live is None:
            return
        from PySide6.QtWidgets import QInputDialog

        items = ["Render box (where the model is computed)", "Whole cutout (nothing subtracted outside the box)"]
        item, ok = QInputDialog.getItem(self, "Export data − model", "Area:", items, 0, False)
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export data − model", "data_minus_model.fits",
                                              "FITS files (*.fits)")
        if path:
            self.export_fits(path, "box" if item == items[0] else "cutout")

    def _photometry_notes(self, r):
        """{name: text}: a source-plane component's OBSERVED flux (intrinsic x mu, in the render box),
        a point's INTRINSIC flux (observed / |mu(theta)|). See photometry.py."""
        zp, notes = self.live.zeropoint, {}

        def both(f):
            return f"{f:.5g}" + (f", m = {flux_to_mag(f, zp):.3f}" if zp is not None and f > 0 else "")

        for name, c in self.live.components.items():
            a = r.amplitudes[name]
            if name in self.live.off:  # (rendered as 0; show what it has)
                a = self.live.amplitude(name) if c.flux_key else a
            if c.kind == "sersic" and c.plane == "source":
                mu = float(self.live.template(name).sum())  # unit intrinsic flux -> its lensed total
                notes[name] = f"observed in the box: {both(a * mu)}  (μ = {mu:.3g})"
            elif c.kind == "point" and name in self.setup.points:
                mu = self.setup.points[name].mu
                notes[name] = f"intrinsic: {both(a / abs(mu))}  (μ(θ) = {mu:+.3g})"
        for name in self.live.off:
            notes[name] = "not in the model now (its \"in model\" box)" + (f"; {notes[name]}" if name in notes else "")
        return notes

    def set_flux_unit(self, unit):
        """Settings: every flux parameter as 'mag' or 'flux' (photometry.convert_fluxes)."""
        if self.live is None:
            return
        try:
            spec, notes = convert_fluxes(self.spec, unit, self.live.zeropoint, self.live.pixscale)
        except ValueError as e:  # SpecError is a ValueError
            self.statusBar().showMessage(f"Settings: {e}", 8000)
            self._show_flux_unit()
            return
        self.spec = spec
        self.render()
        self.statusBar().showMessage(f"fluxes as {'magnitudes' if unit == 'mag' else 'fluxes'}"
                                     + (f" ({'; '.join(notes)})" if notes else ""), 10000)

    def _show_flux_unit(self):
        """Tick the Settings item that matches the spec (magnitudes if any non-sky component uses them)."""
        unit = self.flux_unit()
        for a in self.flux_units.actions():
            a.setChecked(a.data() == unit)
            if a.data() == "mag":
                a.setEnabled(self.live.zeropoint is not None)

    def _fit_fluxes(self):
        """ "Fit fluxes": one NNLS solve for the SELECTED components' free fluxes, the others kept fixed."""
        if self.live is None:
            return
        chosen = self.panel.selected_names()
        if not chosen:
            self.statusBar().showMessage("Fit fluxes: select one or more components first", 5000)
            return
        self.live.set_spec(self.spec)
        try:
            best = self.live.best_fluxes(chosen)
            spec, notes = set_fluxes(self.spec, best, self.live.zeropoint, self.live.pixscale)
        except (ValueError, KeyError) as e:
            self.statusBar().showMessage(f"Fit fluxes: {e}", 8000)
            return
        self.spec = spec
        self.render()
        fitted = [n for n in chosen if n in best]
        off = [n for n in chosen if n in self.live.off]
        kept = [n for n in chosen if n not in best and n not in self.live.off]  # locked or tied fluxes
        text = "fitted the flux of " + (", ".join(fitted) if fitted else "nothing")
        if kept:
            text += f"; kept (locked or tied): {', '.join(kept)}"
        if off:
            text += f"; left out (not in the model): {', '.join(off)}"
        self.statusBar().showMessage(text + (f" ({'; '.join(notes)})" if notes else ""), 8000)

    def _tool_changed(self, *_):
        tool = self.tools.checkedAction().text()
        select = tool == "Select / move"
        for p in (self.source, self.image, self.image2):
            p.drawing = tool == "Draw ellipse"
        for roi in self.rois.values():
            roi.translatable = select
            for h in roi.getHandles():
                h.setVisible(select)
        for handles in self.bend_handles.values():
            for h in handles.values():
                h.movable = select  # (draggable in Select / move only)
        editing = tool == "Edit mask" and self.live is not None
        if editing and self.mask_roi is None:
            self._start_mask_edit()
        if self.mask_roi is not None:
            self.mask_roi.setVisible(editing)
        self.apply_mask_button.setVisible(editing)
        self.revert_mask_button.setVisible(editing)

    # --- the mask (G6c) ------------------------------------------------------------------------

    def mask_theta(self):
        """The mask polygon of the configuration as theta vertices (N, 2), arcsec."""
        radec = np.asarray(self.cfg["mask"]["polygon_radec"], float)
        tx, ty = self.setup.field.sky_to_theta(radec[:, 0], radec[:, 1])
        return np.column_stack([np.asarray(tx, float), np.asarray(ty, float)])

    def _start_mask_edit(self, vertices=None):
        """Show the (current, or given) mask polygon as an editable closed polygon in the image plane."""
        if self.mask_roi is not None:
            self.image.plot.removeItem(self.mask_roi)
        pts = self.mask_theta() if vertices is None else np.asarray(vertices, float)
        self.mask_roi = pg.PolyLineROI([tuple(p) for p in pts], closed=True,
                                       pen=pg.mkPen("#f0a04b", width=1.8), hoverPen=pg.mkPen("#ffd27a", width=2.2))
        self.mask_roi.setZValue(30)
        self.mask_roi.setToolTip("drag a corner; click an edge to add a corner; right-click a corner to remove it; "
                                 "drag inside to move the whole mask")
        self.image.plot.addItem(self.mask_roi)

    def mask_vertices(self):
        """The edited polygon's vertices in theta (N, 2) (the ROI may also have been moved as a whole)."""
        roi = self.mask_roi
        return np.array([[q.x(), q.y()] for q in (roi.mapToParent(p) for p in roi.getState()["points"])])

    def apply_mask(self):
        """Edit mask -> Apply: the polygon (theta) -> RA/Dec -> mask: polygon_radec; rebuild (same model)."""
        if self.mask_roi is None:
            return None
        v = self.mask_vertices()
        if len(v) < 3:
            self.statusBar().showMessage("a mask needs at least 3 corners", 6000)
            return "a mask needs at least 3 corners"
        ra, dec = self.setup.field.theta_to_sky(v[:, 0], v[:, 1])
        mask = dict(copy.deepcopy(self.cfg["mask"]),
                    polygon_radec=[[float(a), float(d)] for a, d in zip(np.atleast_1d(ra), np.atleast_1d(dec))])
        err = self.apply_settings({"mask": mask})
        if err:
            self.statusBar().showMessage(f"mask not applied: {err}", 10000)
            return err
        self._start_mask_edit()  # the new mask, from its RA/Dec
        self._tool_changed()
        self.statusBar().showMessage(f"mask applied: {int(self.live.mask.sum())} fit pixels "
                                     f"(render box {self.live.data.shape[0]} x {self.live.data.shape[1]})", 8000)
        return None

    def revert_mask(self):
        """Edit mask -> Revert: back to the current mask."""
        if self.live is not None:
            self._start_mask_edit()
            self._tool_changed()

    def _value_text(self, panel, row, col):
        """'Data = 0.123 MJy/sr (smoothed 0.101)' of the pixel [row, col] of what ``panel`` shows, or ''."""
        if panel not in self._arrays:
            return ""
        display, raw, shown = self._arrays[panel]
        if not (0 <= row < raw.shape[0] and 0 <= col < raw.shape[1]):
            return ""
        unit = {"Residual": "σ", "Source plane": "(model)"}.get(display, self.setup.img.bunit)
        v = float(raw[row, col])
        text = f"   {display} = {v:.5g} {unit}" if np.isfinite(v) else f"   {display}: no value (outside the mask)"
        if self.smooths[display].on and np.isfinite(shown[row, col]):
            text += f" (smoothed {float(shown[row, col]):.5g})"
        return text

    def pixel_at(self, tx, ty):
        """0-based [row, col] of the render-box pixel at theta (tx, ty)."""
        Ainv, t = self._theta_to_pixel
        col, row = Ainv @ (np.array([tx, ty]) - t)
        return int(round(float(row))), int(round(float(col)))

    def _image_mouse(self, tx, ty, panel="image"):
        """Hover read-out: theta, beta, RA/Dec, and the value of the pixel under the mouse in that panel."""
        if self.setup is None:
            return
        try:
            text = theta_text(self.setup, tx, ty)
        except Exception:  # outside the deflection maps
            text = f"theta = ({tx:+.3f}, {ty:+.3f})\""
        self.coords.setText(text + self._value_text(panel, *self.pixel_at(tx, ty)))

    def _source_mouse(self, bx, by):
        if self.setup is None:
            return
        x0, y0, step = self._src_geom
        row, col = int(np.floor((by - y0) / step)), int(np.floor((bx - x0) / step))
        self.coords.setText(f"beta = ({bx:+.4f}, {by:+.4f})\"" + self._value_text("source", row, col))


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.load(argv[1])
    return app.exec()
