"""Settings -> Model settings… (G6b): cosmology and redshifts, rendering, mask margin, fit.

Shows the configuration's values (with forwardgal's defaults where a key is
missing). OK calls ``apply(sections)`` with the changed sections, e.g.
{"cosmology": {...}, "render": {...}, "mask": {...}, "fit": {...}}; it
returns None (applied: the dialog closes) or an error message (shown).
Non-blocking (``open()``).
"""

from __future__ import annotations

import copy

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from forwardgal.gui.rows import parse_value
from forwardgal.render import RenderSettings

MUTED = "#8a9099"
RENDER = (("subsample", "subsample", 1, 15, "s x s points per pixel"),
          ("patch", "exact patch (unlensed)", 1, 31, "pixels per side, odd: integrated exactly around the centre"),
          ("exact_pixels", "exact pixels (lensed)", 0, 5000, "pixels nearest the centre in the source plane"),
          ("gl_nodes", "Gauss-Legendre nodes", 2, 64, "per angular segment of an exact pixel"),
          ("bend_split", "bend split", 1, 16, "k x k sub-quads of exact pixels of bent components"))


class SettingsDialog(QDialog):
    def __init__(self, cfg, lensed, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model settings")
        self._cfg, self._apply, self._lensed = cfg, apply, lensed
        lay = QVBoxLayout(self)

        box = QGroupBox("Cosmology and redshifts" + ("" if lensed else " (no lens: not used)"))
        form = QFormLayout(box)
        cos = cfg.get("cosmology", {})
        self.cosmo = {}
        for key, label in (("H0", "H0 [km/s/Mpc]"), ("Om0", "Ωm"), ("Ode0", "ΩΛ"), ("z_lens", "z_lens"),
                           ("z_src", "z_src")):
            f = QLineEdit("" if cos.get(key) is None else f"{cos[key]:g}")
            f.setEnabled(lensed)
            form.addRow(label, f)
            self.cosmo[key] = f
        if lensed:
            form.addRow(QLabel(f"<span style='color:{MUTED}'>the maps are for z_ref = "
                               f"{cfg['lens'].get('z_ref')}; another z_src rescales them by "
                               "(D_ls/D_s)(z_src) / (D_ls/D_s)(z_ref)</span>"))
        lay.addWidget(box)

        box = QGroupBox("Rendering")
        form = QFormLayout(box)
        rend = {**RenderSettings().__dict__, **cfg.get("render", {})}
        self.render = {}
        for key, label, lo, hi, tip in RENDER:
            b = QSpinBox()
            b.setRange(lo, hi)
            b.setValue(int(rend[key]))
            b.setToolTip(tip)
            form.addRow(label, b)
            self.render[key] = b
        self.margin = QSpinBox()
        self.margin.setRange(0, 500)
        self.margin.setValue(int(cfg.get("mask", {}).get("margin_px", 35)))
        self.margin.setToolTip("pixels rendered around the mask (the render box); a wide PSF needs more")
        form.addRow("mask margin [px]", self.margin)
        lay.addWidget(box)

        box = QGroupBox("Fit")
        form = QFormLayout(box)
        fit = cfg.get("fit", {})
        self.method = QComboBox()
        self.method.addItems(["lm", "lbfgs"])
        self.method.setCurrentText(fit.get("method", "lm"))
        self.max_iter = QSpinBox()
        self.max_iter.setRange(0, 100000)
        self.max_iter.setSpecialValueText("default")  # 0 -> the method's default (LM 100, L-BFGS 300)
        self.max_iter.setValue(int(fit.get("max_iter") or 0))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(int(fit.get("seed", 0)))
        form.addRow("method", self.method)
        form.addRow("max iterations", self.max_iter)
        form.addRow("seed (jittered starts)", self.seed)
        lay.addWidget(box)

        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        lay.addWidget(buttons)

    def sections(self):
        """The changed sections (only those that differ from the configuration); ValueError for a bad number."""
        cfg, out = self._cfg, {}
        if self._lensed:
            cos = dict(cfg.get("cosmology", {}))
            for key, f in self.cosmo.items():
                cos[key] = parse_value(f.text())
            if cos != cfg.get("cosmology", {}):
                out["cosmology"] = cos
        rend = {k: b.value() for k, b in self.render.items()}
        if rend["patch"] % 2 == 0:
            raise ValueError("the exact patch must have an odd number of pixels per side")
        if rend != {**RenderSettings().__dict__, **cfg.get("render", {})}:
            out["render"] = rend
        if self.margin.value() != int(cfg.get("mask", {}).get("margin_px", 35)):
            out["mask"] = dict(copy.deepcopy(cfg.get("mask", {})), margin_px=self.margin.value())
        fit = dict(cfg.get("fit", {}), method=self.method.currentText(), seed=self.seed.value(),
                   max_iter=self.max_iter.value() or None)
        if fit.get("max_iter") is None:
            fit.pop("max_iter")
        old = dict(cfg.get("fit", {}))
        old.setdefault("method", "lm")
        old.setdefault("seed", 0)
        if fit != old:
            out["fit"] = fit
        return out

    def ok(self):
        try:
            changed = self.sections()
        except ValueError as e:
            self.error.setText(str(e))
            return
        if not changed:
            self.accept()
            return
        err = self._apply(changed)
        if err:
            self.error.setText(err)
            return
        self.accept()
