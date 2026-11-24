"""Settings -> Preferences… : settings of this computer (not of a model file), kept by QSettings.

Now: the memory limit for fits and NUTS runs started from the GUI (GB; 0 = no
limit). A run that goes over it is stopped (the results are never made
coarser to fit: the limit only protects the computer).
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

MUTED = "#8a9099"


class PreferencesDialog(QDialog):
    def __init__(self, memory_gb, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._apply = apply
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.memory = QDoubleSpinBox()
        self.memory.setRange(0.0, 512.0)
        self.memory.setDecimals(2)
        self.memory.setSingleStep(0.5)
        self.memory.setSuffix(" GB")
        self.memory.setSpecialValueText("no limit")  # 0
        self.memory.setValue(float(memory_gb))
        form.addRow("memory limit for fits", self.memory)
        lay.addLayout(form)
        note = QLabel(f"<span style='color:{MUTED}'>A fit or NUTS run started here that uses more is stopped "
                      "(nothing is saved). It measures what Activity Monitor calls Memory (compressed memory "
                      "included). It only protects the computer: it never changes how a fit is computed.</span>")
        note.setWordWrap(True)
        lay.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def ok(self):
        self._apply(self.memory.value())
        self.accept()
