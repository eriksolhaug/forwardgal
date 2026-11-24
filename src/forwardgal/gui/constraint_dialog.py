"""Dialog to edit one parameter's hard bounds and Gaussian prior (G2d-2).

Four optional fields: lo, hi (hard bounds; flat prior between them) and
mu, sigma (Gaussian prior). An empty field means "none". On OK the fields
are handed to ``apply(fields)``, which tries the edit (forwardgal.edit
set_param) and returns None, or an error message: then the dialog stays
open and shows it, so a refused edit never changes the model.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from forwardgal.gui.rows import parse_value

FIELDS = ("lo", "hi", "prior_mu", "prior_sigma")


def parse_optional(text):
    """A field's number, or None if it is empty (only spaces)."""
    return None if not text.strip() else parse_value(text)


class ConstraintDialog(QDialog):
    """Edit lo, hi, prior_mu, prior_sigma of parameter ``title`` (e.g. "disk.n").

    p : the parameter's dict in the spec (for the current settings)
    value_text : its current value, as shown in the card
    apply : callable(fields dict) -> None (accepted) or an error message
    """

    def __init__(self, title, p, value_text, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Constraint on {title}")
        self._apply = apply
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{title}</b>  <span style='color:#8a9099'>current value {value_text}</span>"))
        grid = QGridLayout()
        self.fields = {}
        for row, (name, keys) in enumerate((("Hard bounds", ("lo", "hi")), ("Gaussian prior", ("prior_mu", "prior_sigma")))):
            grid.addWidget(QLabel(name), row, 0)
            for k, key in enumerate(keys):
                grid.addWidget(QLabel({"lo": "lo", "hi": "hi", "prior_mu": "μ", "prior_sigma": "σ"}[key]), row, 1 + 2 * k)
                f = QLineEdit("" if p.get(key) is None else f"{p[key]:g}")
                f.setFixedWidth(90)
                f.setPlaceholderText("none")
                grid.addWidget(f, row, 2 + 2 * k)
                self.fields[key] = f
        lay.addLayout(grid)
        lay.addWidget(QLabel("<span style='color:#8a9099'>empty: none. Bounds are hard (a flat prior inside); "
                             "give μ and σ together.</span>"))
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        clear = QPushButton("Clear")
        clear.setToolTip("remove the bounds and the prior")
        buttons.addButton(clear, QDialogButtonBox.ResetRole)
        clear.clicked.connect(lambda: [f.clear() for f in self.fields.values()])
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self):
        """{field: float or None}; ValueError if a field is not a number."""
        out = {}
        for key, f in self.fields.items():
            try:
                out[key] = parse_optional(f.text())
            except ValueError:
                raise ValueError(f"{key}: not a number: {f.text()!r}") from None
        return out

    def ok(self):
        """Try the edit; close only if it was accepted."""
        try:
            err = self._apply(self.values())
        except ValueError as e:
            err = str(e)
        if err:
            self.error.setText(err)
            return
        self.accept()
