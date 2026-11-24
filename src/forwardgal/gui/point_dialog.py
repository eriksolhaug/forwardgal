"""Point options… (G6d): a point component's position, snap limit, PSF and PSF reference.

Shows what config.build made of it (PointInfo: theta, RA/Dec, snap in pixels, the PSF file, mu).
OK calls ``apply(changes)`` with
    {"ra": deg, "dec": deg, "max_snap": float or None (no limit),
     "psf": None (the band's global PSF) or {"file", "ext", "reference"},
     "global_reference": the global PSF's reference (only with psf None)}
and gets None (applied: the dialog closes) or an error message (shown).
A reference is "centroid", "centre" or [row, col] (stamp pixels, 0-based).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from forwardgal.gui.rows import parse_value

MUTED = "#8a9099"


class ReferenceField(QWidget):
    """centroid / centre / pixel [row, col]."""

    def __init__(self, reference, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.kind = QComboBox()
        self.kind.addItems(["centroid", "centre", "pixel"])
        self.row, self.col = QLineEdit(), QLineEdit()
        for f, tip in ((self.row, "row"), (self.col, "column")):
            f.setFixedWidth(70)
            f.setPlaceholderText(tip)
        if isinstance(reference, str):
            self.kind.setCurrentText(reference)
        else:
            self.kind.setCurrentText("pixel")
            self.row.setText(f"{reference[0]:g}")
            self.col.setText(f"{reference[1]:g}")
        self.kind.currentTextChanged.connect(self._show)
        for w in (self.kind, self.row, self.col):
            h.addWidget(w)
        self._show()

    def _show(self, *_):
        pixel = self.kind.currentText() == "pixel"
        self.row.setVisible(pixel)
        self.col.setVisible(pixel)

    def value(self):
        k = self.kind.currentText()
        return k if k != "pixel" else [parse_value(self.row.text()), parse_value(self.col.text())]


class PointDialog(QDialog):
    def __init__(self, name, comp, info, global_psf, apply, parent=None):
        """comp: the point's spec dict (ra/dec or x0/y0, max_snap, psf); info: its PointInfo;
        global_psf: the configuration's psf section."""
        super().__init__(parent)
        self.setWindowTitle(f"Point options: {name}")
        self._apply = apply
        lay = QVBoxLayout(self)
        now = QLabel(f"<b>{name}</b> <span style='color:{MUTED}'>now: θ = ({info.theta[0]:+.4f}, "
                     f"{info.theta[1]:+.4f})″, snapped by {info.snap:.3f} px, μ = {info.mu:+.3g}<br>"
                     f"PSF {info.psf}</span>")
        now.setWordWrap(True)  # (long PSF paths)
        now.setMaximumWidth(640)
        lay.addWidget(now)
        form = QFormLayout()
        self.ra, self.dec = QLineEdit(f"{info.radec[0]:.8f}"), QLineEdit(f"{info.radec[1]:.8f}")
        form.addRow("RA [deg]", self.ra)
        form.addRow("Dec [deg]", self.dec)
        snap = comp.get("max_snap", 0.1)
        self.no_limit = QCheckBox("no limit (trust the PSF)")
        self.max_snap = QDoubleSpinBox()
        self.max_snap.setDecimals(3)
        self.max_snap.setRange(0.0, 5.0)
        self.max_snap.setSingleStep(0.05)
        self.max_snap.setValue(0.1 if snap is None else float(snap))
        self.no_limit.setChecked(snap is None)
        self.no_limit.toggled.connect(lambda on: self.max_snap.setEnabled(not on))
        self.max_snap.setEnabled(snap is not None)
        row = QHBoxLayout()
        row.addWidget(self.max_snap)
        row.addWidget(QLabel("px"))
        row.addWidget(self.no_limit)
        form.addRow("snap limit", row)
        lay.addLayout(form)

        own = comp.get("psf")
        self.use_global = QRadioButton(f"the band's PSF ({global_psf['file'].split('/')[-1]})")
        self.use_own = QRadioButton("its own PSF file:")
        self.use_global.setChecked(own is None)
        self.use_own.setChecked(own is not None)
        lay.addWidget(self.use_global)
        row = QHBoxLayout()
        row.addWidget(self.use_own)
        self.file = QLineEdit("" if own is None else str(own.get("file", "")))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.ext = QSpinBox()
        self.ext.setRange(0, 20)
        self.ext.setValue(int((own or {}).get("ext", 0)))
        for w in (self.file, browse, QLabel("ext"), self.ext):
            row.addWidget(w)
        lay.addLayout(row)
        form = QFormLayout()
        ref = (own or {}).get("reference", "centroid") if own is not None else global_psf.get("reference", "centroid")
        self.reference = ReferenceField(ref)
        form.addRow("PSF reference", self.reference)
        lay.addLayout(form)
        lay.addWidget(QLabel(f"<span style='color:{MUTED}'>reference: where in the stamp the point lies "
                             "(centroid: a PSF made for this image; centre: GALFIT's). For the band's PSF it is "
                             "the psf: reference of every point that uses it.</span>"))
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        lay.addWidget(buttons)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "PSF for this point", "", "FITS files (*.fits *.fit *.fits.gz)")
        if path:
            self.file.setText(path)
            self.use_own.setChecked(True)

    def changes(self):
        """The dialog's values (see the module docstring); ValueError for a bad entry."""
        out = {"ra": parse_value(self.ra.text()), "dec": parse_value(self.dec.text()),
               "max_snap": None if self.no_limit.isChecked() else float(self.max_snap.value())}
        ref = self.reference.value()
        if self.use_own.isChecked():
            if not self.file.text().strip():
                raise ValueError("choose the point's PSF file (or use the band's PSF)")
            out["psf"] = {"file": self.file.text().strip(), "ext": self.ext.value(), "reference": ref}
        else:
            out["psf"], out["global_reference"] = None, ref
        return out

    def ok(self):
        try:
            changes = self.changes()
        except ValueError as e:
            self.error.setText(str(e))
            return
        err = self._apply(changes)
        if err:
            self.error.setText(err)
            return
        self.accept()
