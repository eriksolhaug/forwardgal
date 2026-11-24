"""Scale Parameters (D5), as in DS9: the pixel histogram of one display, with the limits as a
draggable band (and two fields); Apply sets them as "user" limits (in data units).

``apply(lo, hi)`` returns None (accepted) or an error message (the dialog stays open and shows it).
Non-blocking: the window calls ``open()``.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout

from forwardgal.gui.rows import parse_value


class ScaleDialog(QDialog):
    def __init__(self, display, sample, lo, hi, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Scale Parameters: {display}")
        self.resize(560, 360)
        self._apply = apply
        s = np.asarray(sample, float)
        s = s[np.isfinite(s)]
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{display}</b> <span style='color:#8a9099'>pixel values in the mask "
                             "(log counts); drag the band or its edges, or type the limits</span>"))
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(y=True)
        a, b = np.percentile(s, [0.1, 99.9]) if s.size else (0.0, 1.0)
        a, b = min(a, lo), max(b, hi)
        if not b > a:
            b = a + 1.0
        counts, edges = np.histogram(s, bins=200, range=(a, b))
        self.plot.plot(edges, np.maximum(counts, 0.5), stepMode="center", fillLevel=None,
                       pen=pg.mkPen("#c9cdd3"))
        self.region = pg.LinearRegionItem(values=(lo, hi), brush=pg.mkBrush(240, 160, 75, 50))
        self.plot.addItem(self.region)
        lay.addWidget(self.plot, 1)
        row = QHBoxLayout()
        self.lo, self.hi = QLineEdit(f"{lo:.6g}"), QLineEdit(f"{hi:.6g}")
        for label, field in (("low", self.lo), ("high", self.hi)):
            row.addWidget(QLabel(label))
            field.setFixedWidth(120)
            field.editingFinished.connect(self._fields_to_region)
            row.addWidget(field)
        row.addStretch(1)
        lay.addLayout(row)
        self.region.sigRegionChanged.connect(self._region_to_fields)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.ok)
        buttons.rejected.connect(self.reject)
        self.apply_button = buttons.button(QDialogButtonBox.Apply)
        lay.addWidget(buttons)

    def _region_to_fields(self):
        lo, hi = self.region.getRegion()
        self.lo.setText(f"{lo:.6g}")
        self.hi.setText(f"{hi:.6g}")

    def _fields_to_region(self):
        try:
            lo, hi = parse_value(self.lo.text()), parse_value(self.hi.text())
        except ValueError:
            return
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)

    def ok(self):
        """Apply the limits (the dialog stays open, to try others)."""
        try:
            lo, hi = parse_value(self.lo.text()), parse_value(self.hi.text())
        except ValueError as e:
            self.error.setText(str(e))
            return
        err = self._apply(lo, hi)
        self.error.setText(err or "")
