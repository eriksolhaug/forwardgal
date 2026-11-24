"""Smooth Parameters (as in DS9): function, radius (pixels), Gaussian sigma; Apply turns smoothing on.

``apply(Smooth)`` returns None (accepted) or an error message (shown; the dialog stays open). Non-blocking.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from forwardgal.gui.constraint_dialog import parse_optional
from forwardgal.smoothing import FUNCTIONS, Smooth

MUTED = "#8a9099"


class SmoothDialog(QDialog):
    def __init__(self, display, current, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Smooth Parameters: {display}")
        self._apply = apply
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{display}</b> <span style='color:{MUTED}'>display only: the fit never "
                             "sees it; normalized over the pixels with values (the mask)</span>"))
        g = QGridLayout()
        self.function = QComboBox()
        self.function.addItems(FUNCTIONS)
        self.function.setCurrentText(current.function)
        self.radius = QSpinBox()
        self.radius.setRange(1, 50)
        self.radius.setValue(current.radius)
        self.sigma = QLineEdit("" if current.sigma is None else f"{current.sigma:g}")
        self.sigma.setPlaceholderText("radius / 2")
        for i, (label, w, unit) in enumerate((("function", self.function, "boxcar: square · tophat: disk"),
                                              ("radius", self.radius, "pixels (kernel 2r + 1 wide)"),
                                              ("sigma", self.sigma, "pixels (gaussian only)"))):
            g.addWidget(QLabel(label), i, 0)
            g.addWidget(w, i, 1)
            g.addWidget(QLabel(f"<span style='color:{MUTED}'>{unit}</span>"), i, 2)
        lay.addLayout(g)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        self.apply_button = buttons.button(QDialogButtonBox.Apply)
        self.apply_button.clicked.connect(self.ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def ok(self):
        try:
            s = Smooth(on=True, function=self.function.currentText(), radius=self.radius.value(),
                       sigma=parse_optional(self.sigma.text()))
        except ValueError as e:
            self.error.setText(str(e))
            return
        self.error.setText(self._apply(s) or "")
