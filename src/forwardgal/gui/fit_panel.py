"""The fit card (G4), after the mockup: "Fit selected", "Fit all (multi-start)", progress, Cancel;
"Run NUTS -> posteriors" with warm-up, samples and chains (``nuts_clicked(warmup, samples, chains)``;
enabled by the window only while an unchanged fit is loaded) and "Open corner plot".

It only shows buttons and text: ``fit_clicked(selected)`` (True: only the
selected components) and ``cancel_clicked()`` go to the main window, which
runs the fit in a separate process (forwardgal.run) and calls
``set_running`` / ``set_progress``.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

MUTED = "#8a9099"


class FitPanel(QWidget):
    fit_clicked = Signal(bool)  # True: "Fit selected"
    cancel_clicked = Signal()
    nuts_clicked = Signal(int, int, int)  # warm-up, samples, chains
    corner_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.fit_selected = QPushButton("Fit selected")
        self.fit_selected.setToolTip("fit only the selected components (the others are held at their values)")
        self.fit_all = QPushButton("Fit all (multi-start)")
        self.fit_all.setToolTip("fit every free parameter, from several starts (the best one is kept)")
        self.fit_selected.clicked.connect(lambda: self.fit_clicked.emit(True))
        self.fit_all.clicked.connect(lambda: self.fit_clicked.emit(False))
        row.addWidget(self.fit_selected)
        row.addWidget(self.fit_all)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("starts"))
        self.starts = QSpinBox()
        self.starts.setRange(1, 50)
        self.starts.setValue(5)
        self.starts.setToolTip("number of starts (the first from the current values, the others jittered)")
        row.addWidget(self.starts)
        row.addStretch(1)
        self.cancel = QPushButton("Cancel")
        self.cancel.setToolTip("stop the running fit (nothing is saved)")
        self.cancel.clicked.connect(self.cancel_clicked.emit)
        row.addWidget(self.cancel)
        lay.addLayout(row)
        self.nuts = QPushButton("Run NUTS → posteriors")
        self.nuts.setObjectName("accent")
        self.nuts.clicked.connect(lambda: self.nuts_clicked.emit(self.warmup.value(), self.samples.value(),
                                                                 self.chains.value()))
        lay.addWidget(self.nuts)
        row = QHBoxLayout()
        self.warmup, self.samples, self.chains = QSpinBox(), QSpinBox(), QSpinBox()
        for label, box, lo, hi, val, tip in (
                ("warm-up", self.warmup, 10, 20000, 500, "NUTS warm-up steps per chain (step-size tuning)"),
                ("samples", self.samples, 10, 100000, 1000, "posterior samples per chain"),
                ("chains", self.chains, 1, 8, 2, "chains (run one after another; R-hat needs 2 or more)")):
            box.setRange(lo, hi)
            box.setValue(val)
            box.setToolTip(tip)
            row.addWidget(QLabel(label))
            row.addWidget(box)
        row.addStretch(1)
        lay.addLayout(row)
        self.corner = QPushButton("Open corner plot")
        self.corner.setToolTip("the last NUTS run's corner plot (with the Laplace 1-sigma ellipses)")
        self.corner.clicked.connect(self.corner_clicked.emit)
        self.corner.setEnabled(False)
        lay.addWidget(self.corner)
        self.progress = QLabel("")
        self.progress.setObjectName("mono")
        self.progress.setStyleSheet(f"color: {MUTED};")
        self.progress.setWordWrap(True)
        lay.addWidget(self.progress)
        note = QLabel(f"<span style='color:{MUTED}'>the model is saved to the history first; the fit appears "
                      "there as a new row and is loaded when it is done</span>")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.set_running(False)

    def set_running(self, running):
        for b in (self.fit_selected, self.fit_all, self.starts, self.warmup, self.samples, self.chains):
            b.setEnabled(not running)
        self.cancel.setEnabled(running)
        if running:
            self.nuts.setEnabled(False)

    def set_nuts_enabled(self, enabled, why=""):
        """NUTS starts at a fit: the window enables it only while an unchanged fit is loaded."""
        self.nuts.setEnabled(enabled)
        self.nuts.setToolTip("NUTS posteriors around the loaded fit (a new history row; long: tens of minutes "
                             "on 1A with the defaults)" if enabled else why)

    def set_progress(self, text):
        self.progress.setText(text)
