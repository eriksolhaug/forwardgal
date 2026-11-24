"""The components column of the main window (G2b), after the mockup.

Two cards:
  - the component list: colour dot (the colour of the component's ellipse),
    name, short summary ("n 3.80 · source");
  - the selected component's parameters: label | value | constraint | lock,
    one row per gui.rows.Row, then its flux.

Values can be typed (G2c): pressing Enter or leaving a changed field emits
``edited(name, key, text)``; the main window turns it into a checked spec
edit, or calls ``revert`` if the edit is refused. Clicking a lock box
(G2d; enabled only where Row.lockable) emits ``lock_changed(name, key,
locked)``, handled the same way. Clicking a constraint (G2d-2; same rule)
emits ``constraint_clicked(name, key)``: the window opens its dialog.
Under the rows (G2d-3): this component's ties, each with an "untie" button
(``untie_clicked(target)``), ties to it from others (for information), and
"+ Tie…" (``tie_clicked(name)``).
Several components can be selected (A4): Ctrl-click (⌘-click on a Mac;
Qt maps Ctrl to ⌘ there) toggles one, Shift-click selects a range,
Ctrl/⌘+A all; the card shows the one clicked last (``current``), and
``selection_changed()`` is emitted. "Fit fluxes" (``fit_fluxes_clicked``)
is for the selected components; its label says how many.
Each list row starts with a narrow "show marker" box (V1): unticking it
hides that component's ellipse / point marker (display only: the model is
not changed). ``hidden`` is the set of hidden names; ``set_hidden``
changes it; both emit ``visibility_changed()``. The sky has no marker.
A second box (M1), "in model", leaves a component out of the model for
now (``off``, ``set_off``; emits ``model_changed()``): the window renders
without it and Fit fluxes treats it as absent. The spec keeps it.
"+ mode" / "− mode" on a Sersic's card (G6a) emit ``mode_add(name, kind, m)`` /
``mode_remove(name, kind, m)`` (kind "fourier" or "bending").
"+ Add Sérsic" in the list's header emits ``add_clicked()``; "Remove" on
the card emits ``remove_clicked(name)`` (A3a): the window edits the spec.
``set_model`` is called after every render; widgets are rebuilt only when
the structure (component names, or the selected component's rows) changes,
otherwise only their texts change (never the field being typed in).
"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from forwardgal.gui.rows import param_rows, summary

MUTED = "#8a9099"


def _dot(colour):
    d = QLabel()
    d.setFixedSize(10, 10)
    d.setStyleSheet(f"background: {colour}; border-radius: 5px;")
    return d


def _clear(layout):
    """Remove everything from a layout, also from layouts inside it.

    Widgets are hidden at once (deleteLater alone leaves them on screen
    until the next pass of the event loop) and then deleted.
    """
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


class ComponentsPanel(QWidget):
    """Component list + parameter card. Emits ``selected(name)`` when the user picks a component."""

    selected = Signal(str)
    edited = Signal(str, str, str)  # component name, row key, the text typed
    lock_changed = Signal(str, str, bool)  # component name, row key, True = lock (fix) it
    constraint_clicked = Signal(str, str)  # component name, row key
    tie_clicked = Signal(str)  # component name ("+ Tie…")
    untie_clicked = Signal(str)  # the tied parameter, "component.key"
    fit_fluxes_clicked = Signal()  # "Fit fluxes": one NNLS solve for the selected components' free fluxes
    selection_changed = Signal()  # the set of selected components changed (selected_names())
    visibility_changed = Signal()  # a "show marker" box changed (``hidden``)
    model_changed = Signal()  # an "in model" box changed (``off``)
    mode_add = Signal(str, str, int)  # component, "fourier" / "bending", m
    mode_remove = Signal(str, str, int)
    point_options_clicked = Signal(str)  # a point's "Point options…" (G6d)
    add_clicked = Signal()  # "+ Add Sérsic"
    remove_clicked = Signal(str)  # "Remove" on the card: component name

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Card 1: the list.
        card = QWidget()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Components</b>"))
        head.addStretch(1)
        self.fit_fluxes = QPushButton("Fit fluxes")
        self.fit_fluxes.setToolTip("set the SELECTED components' fluxes to their best values for the current "
                                   "shapes (one linear solve; the others, and locked or tied fluxes, are kept). "
                                   "Select several with Ctrl-click (⌘-click on a Mac), all with Ctrl/⌘+A")
        self.fit_fluxes.clicked.connect(self.fit_fluxes_clicked.emit)
        head.addWidget(self.fit_fluxes)
        self.add_button = QPushButton("+ Add Sérsic")
        self.add_button.setToolTip("a new source-plane Sérsic at the centre of the source-plane view; "
                                   "its flux is fitted once to the light the others leave over")
        self.add_button.clicked.connect(self.add_clicked.emit)
        head.addWidget(self.add_button)
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setObjectName("complist")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        lay.addWidget(self.list)
        outer.addWidget(card)

        # Card 2: the selected component's parameters.
        card = QWidget()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        head = QHBoxLayout()
        self.title = QLabel("")
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(QLabel(f"<span style='color:{MUTED}'>value · constraint · lock</span>"))
        lay.addLayout(head)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setColumnStretch(2, 1)
        lay.addLayout(self.grid)
        self.photometry = QLabel("")  # observed / intrinsic of the selected component (F4b)
        self.photometry.setObjectName("mono")
        self.photometry.setStyleSheet(f"color: {MUTED};")
        self.photometry.setWordWrap(True)
        lay.addWidget(self.photometry)
        self.ties = QVBoxLayout()  # the ties lines (rebuilt when they change)
        self.ties.setSpacing(2)
        lay.addLayout(self.ties)
        self.tie_button = QPushButton("+ Tie…")
        self.tie_button.setToolTip("make a parameter follow another component's (same plane), plus an offset")
        self.tie_button.clicked.connect(lambda: self.tie_clicked.emit(self.current))
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("remove this component (and any ties to or from it)")
        self.remove_button.clicked.connect(lambda: self.remove_clicked.emit(self.current))
        self.add_mode_button, self.remove_mode_button = QPushButton("+ mode"), QPushButton("− mode")
        self.add_mode_menu, self.remove_mode_menu = QMenu(self), QMenu(self)
        self.add_mode_button.setMenu(self.add_mode_menu)
        self.remove_mode_button.setMenu(self.remove_mode_menu)
        self.add_mode_button.setToolTip("add a Fourier (m = 1, 3, 4) or bending (m = 2-5) mode, at zero amplitude")
        self.add_mode_menu.triggered.connect(lambda a: self.mode_add.emit(self.current, *a.data()))
        self.remove_mode_menu.triggered.connect(lambda a: self.mode_remove.emit(self.current, *a.data()))
        self.point_button = QPushButton("Point options…")
        self.point_button.setToolTip("position (RA/Dec), snap limit, PSF and PSF reference of this point")
        self.point_button.clicked.connect(lambda: self.point_options_clicked.emit(self.current))
        tie_row = QHBoxLayout()
        tie_row.addWidget(self.remove_button)
        tie_row.addWidget(self.point_button)
        tie_row.addWidget(self.add_mode_button)
        tie_row.addWidget(self.remove_mode_button)
        tie_row.addStretch(1)
        tie_row.addWidget(self.tie_button)
        lay.addLayout(tie_row)
        outer.addWidget(card)
        outer.addStretch(1)

        self.names = []  # component names in list order
        self.current = None  # the shown component
        self._summaries = {}  # name -> summary QLabel
        self._cells = {}  # row key -> (value QLineEdit, constraint QLabel, lock QCheckBox)
        self._shown_keys = None  # (name, row keys) the grid was built for
        self._model = None  # (spec, values, amplitudes)
        self._texts = {}  # row key -> value text as last shown
        self._ties_shown = None  # (name, ties lines) the ties area was built for
        self._notes = {}  # name -> text under the card
        self.hidden = set()  # names whose marker (ellipse, +) is hidden
        self._vis = {}  # name -> its "show marker" QCheckBox
        self.off = set()  # names left out of the model for now
        self._inmodel = {}  # name -> its "in model" QCheckBox

    # --- data in ------------------------------------------------------------------

    def set_model(self, spec, values, amplitudes, colours, notes=None):
        """Show ``spec`` with current ``values`` ({name: {param: value}}) and ``amplitudes``.

        colours : {name: colour string} (the ellipses' colours)
        notes : {name: text} shown under the card (e.g. observed / intrinsic flux)
        """
        self._model = (spec, values, amplitudes)
        self._notes = notes or {}
        comps = spec["components"]
        names = [c["name"] for c in comps]
        if names != self.names:
            keep = [n for n in self.selected_names() if n in names]  # the selection survives a rebuild
            self._build_list(comps, colours)
            if self.current in names:
                self._set_selection(keep or [self.current])
        for c in comps:
            self._summaries[c["name"]].setText(summary(c, values[c["name"]]))
        if self.current not in names:
            self.current = names[0] if names else None
            self._select_row(self.current)
        self._show(self.current)

    def revert(self):
        """Show the current values again (after a refused edit)."""
        for value, *_ in self._cells.values():
            value.clearFocus()
        self._show(self.current, force=True)

    def select(self, name, add=False):
        """Show component ``name`` (e.g. its ellipse was clicked). Does not emit ``selected``.

        add=False: select only it. add=True (Ctrl/⌘-click): toggle it in the selection, keep the others.
        """
        if name not in self.names:
            return
        if add or name != self.current or self.selected_names() != [name]:
            self.current = name
            self._select_row(name, add)
            self._show(name)

    def selected_names(self):
        """The selected components, in list order."""
        rows = sorted(i.row() for i in self.list.selectedIndexes())
        return [self.names[r] for r in rows if r < len(self.names)]

    # --- building ---------------------------------------------------------------------

    def _build_list(self, comps, colours):
        self.list.blockSignals(True)
        self.list.clear()
        self._summaries, self._vis, self._inmodel = {}, {}, {}
        self.hidden &= {c["name"] for c in comps}  # forget removed components
        self.off &= {c["name"] for c in comps}
        for c in comps:
            w = QWidget()
            w.setObjectName("compitem")  # transparent (app stylesheet): the selection frame shows through
            h = QHBoxLayout(w)
            h.setContentsMargins(6, 4, 6, 4)
            vis = QCheckBox()
            vis.setObjectName("vis")  # narrow, tall indicator (app stylesheet)
            vis.setChecked(c["name"] not in self.hidden)
            vis.setToolTip("show this component's marker (ellipse or +) in the plots")
            vis.clicked.connect(lambda shown, n=c["name"]: self._vis_clicked(n, shown))
            if c["kind"] == "sky":  # no marker: keep the space so the rows line up
                policy = vis.sizePolicy()
                policy.setRetainSizeWhenHidden(True)
                vis.setSizePolicy(policy)
                vis.hide()
            self._vis[c["name"]] = vis
            h.addWidget(vis)
            inm = QCheckBox()
            inm.setObjectName("inmodel")  # narrow, tall, orange (app stylesheet)
            inm.setChecked(c["name"] not in self.off)
            inm.setToolTip("include this component in the model (untick: the model and residual without it, "
                           "for now; Fit fluxes leaves it out; it stays in the model file)")
            inm.clicked.connect(lambda on, n=c["name"]: self._inmodel_clicked(n, on))
            self._inmodel[c["name"]] = inm
            h.addWidget(inm)
            h.addWidget(_dot(colours.get(c["name"], MUTED)))
            h.addWidget(QLabel(c["name"]))
            h.addStretch(1)
            s = QLabel("")
            s.setObjectName("mono")
            s.setStyleSheet(f"color: {MUTED};")
            h.addWidget(s)
            self._summaries[c["name"]] = s
            item = QListWidgetItem()
            item.setSizeHint(w.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, w)
        self.names = [c["name"] for c in comps]
        self.list.setFixedHeight(min(260, 6 + sum(self.list.sizeHintForRow(i) for i in range(self.list.count()))))
        self.list.blockSignals(False)
        self._shown_keys = None

    def _set_selection(self, chosen):
        """Select exactly ``chosen`` (names), keeping ``current`` as the list's current row."""
        self.list.blockSignals(True)
        for i, n in enumerate(self.names):
            self.list.item(i).setSelected(n in chosen)
        if self.current in self.names:
            self.list.setCurrentRow(self.names.index(self.current), QItemSelectionModel.NoUpdate)
        self.list.blockSignals(False)
        self._selection_changed()

    def _select_row(self, name, add=False):
        self.list.blockSignals(True)  # set by the program: no ``selected`` signal
        cmd = QItemSelectionModel.Toggle if add else QItemSelectionModel.ClearAndSelect
        self.list.setCurrentRow(self.names.index(name) if name in self.names else -1, cmd)
        self.list.blockSignals(False)
        self._selection_changed()

    def _show(self, name, force=False):
        if name is None or self._model is None:
            return
        spec, values, amps = self._model
        rows = param_rows(spec, name, values[name], amps)
        keys = (name, tuple(r.key for r in rows))
        if keys != self._shown_keys:
            self._build_grid(rows)
            self._shown_keys = keys
            self.title.setText(f"<b>{name}</b>")
        self._show_ties(name, spec, rows)
        self._show_mode_menus(next(c for c in spec["components"] if c["name"] == name))
        self.photometry.setText(self._notes.get(name, ""))
        self._texts = {r.key: r.text for r in rows}  # as shown: an edit is a change of these
        for r in rows:
            value, constraint, lock = self._cells[r.key]
            if force or not value.hasFocus():  # never overwrite what the user is typing
                value.setText(r.text)
            value.setReadOnly(not r.editable)  # may change without a rebuild (e.g. a new tie)
            value.setToolTip(r.tooltip)
            constraint.setText(r.constraint or ("none" if r.lockable else ""))
            constraint.setEnabled(r.lockable)  # editable where the lock is (not tied, joint, point, flux)
            lock.setChecked(r.locked)  # from the program: does not emit ``clicked``
            lock.setEnabled(r.lockable)  # may change without a rebuild (e.g. a new tie)
            lock.setToolTip(r.tooltip or ("locked: not varied by the fit" if r.locked else
                                          "free: varied by the fit (click to lock it at its value)"))

    def _show_mode_menus(self, comp):
        """The "+ mode" / "− mode" menus: the modes this Sersic can get / has (hidden for other kinds)."""
        from forwardgal.components import BEND_M, FOURIER_M

        sersic = comp["kind"] == "sersic"
        p = comp["params"]
        have = [("fourier", m) for m in FOURIER_M if f"c{m}" in p] + [("bending", m) for m in BEND_M if f"b{m}" in p]
        can = [("fourier", m) for m in FOURIER_M if f"c{m}" not in p and f"a{m}" not in p] + \
              [("bending", m) for m in BEND_M if f"b{m}" not in p]
        for menu, items in ((self.add_mode_menu, can), (self.remove_mode_menu, have)):
            menu.clear()
            for kind, m in items:
                a = menu.addAction(f"{'Fourier' if kind == 'fourier' else 'bending'} m = {m}")
                a.setData((kind, m))
        self.point_button.setVisible(comp["kind"] == "point")
        self.add_mode_button.setVisible(sersic)
        self.remove_mode_button.setVisible(sersic)
        self.add_mode_button.setEnabled(bool(can))
        self.remove_mode_button.setEnabled(bool(have))

    def _show_ties(self, name, spec, rows):
        """The ties lines of component ``name``: its own (with "untie"), and others' ties to it."""
        ties = spec.get("ties", {})
        labels = {r.key: r.label for r in rows}
        own = [(t, s, o) for t, (s, o) in ties.items() if t.split(".")[0] == name]
        to_me = [(t, s) for t, (s, _) in ties.items() if s.split(".")[0] == name]
        key = (name, tuple(own), tuple(to_me))
        self.tie_button.setEnabled(any(r.lockable for r in rows))  # something here may be tied
        if key == self._ties_shown:
            return
        self._ties_shown = key
        _clear(self.ties)
        for target, source, off in own:
            k = target.split(".", 1)[1]
            h = QHBoxLayout()
            text = QLabel(f"{labels.get(k, k)} = {source} {'+' if off >= 0 else '-'} {abs(off):g}")
            text.setObjectName("mono")
            untie = QPushButton("untie")
            untie.setToolTip(f"{target} becomes a parameter of its own again (at its current value)")
            untie.clicked.connect(lambda _=False, t=target: self.untie_clicked.emit(t))
            h.addWidget(QLabel("Tie:"))
            h.addWidget(text)
            h.addStretch(1)
            h.addWidget(untie)
            self.ties.addLayout(h)
        for target, source in to_me:
            note = QLabel(f"{target} follows {name}.{source.split('.', 1)[1]}")
            note.setStyleSheet(f"color: {MUTED};")
            self.ties.addWidget(note)

    def _build_grid(self, rows):
        _clear(self.grid)
        self._cells = {}
        for i, r in enumerate(rows):
            value = QLineEdit()
            value.setReadOnly(not r.editable)
            value.setFixedWidth(110)
            value.editingFinished.connect(lambda key=r.key: self._value_entered(key))
            value.setAlignment(Qt.AlignRight)
            constraint = QPushButton("")
            constraint.setObjectName("constraint")  # flat, left-aligned text (app stylesheet)
            constraint.setCursor(Qt.PointingHandCursor)
            constraint.clicked.connect(lambda _=False, key=r.key: self.constraint_clicked.emit(self.current, key))
            lock = QCheckBox()
            lock.clicked.connect(lambda checked, key=r.key: self.lock_changed.emit(self.current, key, checked))
            value.setToolTip(r.tooltip)
            constraint.setToolTip(r.tooltip or "click to set bounds or a Gaussian prior")
            self.grid.addWidget(QLabel(r.label), i, 0)
            self.grid.addWidget(value, i, 1)
            self.grid.addWidget(constraint, i, 2)
            self.grid.addWidget(lock, i, 3)
            self._cells[r.key] = (value, constraint, lock)

    # --- user input ---------------------------------------------------------------------

    def _row_changed(self, row):
        if 0 <= row < len(self.names):
            self.current = self.names[row]
            self._show(self.current)
            self.selected.emit(self.current)

    def set_hidden(self, names):
        """Hide exactly the markers of ``names`` (e.g. View -> Hide all markers); emits visibility_changed."""
        self.hidden = set(names) & set(self.names)
        for n, box in self._vis.items():
            box.setChecked(n not in self.hidden)
        self.visibility_changed.emit()

    def set_off(self, names):
        """Leave exactly ``names`` out of the model; emits model_changed."""
        self.off = set(names) & set(self.names)
        for n, box in self._inmodel.items():
            box.setChecked(n not in self.off)
        self.model_changed.emit()

    def _inmodel_clicked(self, name, on):
        (self.off.discard if on else self.off.add)(name)
        self.model_changed.emit()

    def _vis_clicked(self, name, shown):
        (self.hidden.discard if shown else self.hidden.add)(name)
        self.visibility_changed.emit()

    def _selection_changed(self):
        n = len(self.selected_names())
        self.fit_fluxes.setText("Fit flux" if n == 1 else f"Fit fluxes ({n})")
        self.fit_fluxes.setEnabled(n > 0)
        self.selection_changed.emit()

    def _value_entered(self, key):
        value = self._cells[key][0]
        text = value.text()
        if not value.isReadOnly() and text != self._texts.get(key):
            self.edited.emit(self.current, key, text)
