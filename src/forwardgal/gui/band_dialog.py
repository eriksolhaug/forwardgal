"""The "Use in band..." dialog (G5): seed another band from a fit (forwardgal.seed.seed_into).

    target band  [F150W ▾]   (the open bands, or "a model file...")
    ranges around this fit's best values (blank = no range: free within the usual bounds):
        position ±[0.02] arcsec (a circle)   R_e ±[20] %   n ±[0.5]   q ±[0.05]   PA ±[10] deg

On OK, ``apply(target_index_or_None, ranges)`` runs (None: choose a model
file); it returns None (done) or an error message (the dialog stays open).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from forwardgal.gui.rows import parse_value
from forwardgal.seed import DEFAULT_RANGES

MUTED = "#8a9099"
# (key in the ranges, label, unit, how the field maps to the range)
FIELDS = (("position", "position ±", "″ (a circle around the best centre)", "abs"),
          ("R_e", "R_e ±", "%", "rel%"), ("n", "n ±", "", "abs"), ("q", "q ±", "", "abs"),
          ("pa_deg", "PA ±", "°", "abs"))


def _text(spec, how):
    if spec is None:
        return ""
    if how == "rel%":
        return f"{100 * spec['rel']:g}"
    return f"{spec:g}"


class UseInBandDialog(QDialog):
    def __init__(self, source, folder, targets, apply, ranges=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Use {folder} in another band")
        self._apply = apply
        ranges = DEFAULT_RANGES["*"] if ranges is None else ranges
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Seed another band from <b>{source}</b> · {folder}: its shapes start at this fit's "
                             "best values,<br>each within a range of them; fluxes are solved afresh "
                             f"<span style='color:{MUTED}'>(the band's own data, PSF, mask, ... are kept)</span>"))
        self.target = QComboBox()
        for t in targets:
            self.target.addItem(t)
        self.target.addItem("a model file…")
        g = QGridLayout()
        g.addWidget(QLabel("target band"), 0, 0)
        g.addWidget(self.target, 0, 1, 1, 2)
        self.fields = {}
        for i, (key, label, unit, how) in enumerate(FIELDS, start=1):
            f = QLineEdit(_text(ranges.get(key), how))
            f.setFixedWidth(80)
            f.setPlaceholderText("no range")
            g.addWidget(QLabel(label), i, 0)
            g.addWidget(f, i, 1)
            g.addWidget(QLabel(f"<span style='color:{MUTED}'>{unit}</span>"), i, 2)
            self.fields[key] = (f, how)
        lay.addLayout(g)
        lay.addWidget(QLabel(f"<span style='color:{MUTED}'>blank: no range (free within the usual bounds)</span>"))
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        lay.addWidget(buttons)

    def ranges(self):
        """{"*": {...}} from the fields (blank fields left out); ValueError for a bad number."""
        out = {}
        for key, (f, how) in self.fields.items():
            if not f.text().strip():
                continue
            x = parse_value(f.text())
            if not x > 0:
                raise ValueError(f"{key}: a range must be positive")
            out[key] = {"rel": x / 100.0} if how == "rel%" else x
        return {"*": out}

    def ok(self):
        try:
            ranges = self.ranges()
        except ValueError as e:
            self.error.setText(str(e))
            return
        i = self.target.currentIndex()
        err = self._apply(None if i == self.target.count() - 1 else i, ranges)
        if err:
            self.error.setText(err)
            return
        self.accept()
