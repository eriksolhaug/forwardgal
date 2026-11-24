"""Dialog to tie one parameter of a component to another component's (G2d-3).

    [parameter of this component] = [source "component.key"] + [offset]

The source list comes from ``candidates(key)`` (edit.tie_candidates: same
key, other component, same plane, not itself tied). The offset starts at
``current_offset(key, source)`` = target - source now, so tying does not
move anything; 0 puts the two exactly together. On OK, ``apply(key, source,
offset)`` tries the edit and returns None or an error message (the dialog
then stays open and shows it).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from forwardgal.gui.rows import parse_value


class TieDialog(QDialog):
    """Tie ``name.<key>`` to another component's same parameter.

    targets : [(key, label)] of this component's parameters that may be tied
    candidates : key -> ["component.key", ...]
    current_offset : (key, source) -> float (target - source now)
    apply : (key, source, offset) -> None (accepted) or an error message
    """

    def __init__(self, name, targets, candidates, current_offset, apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tie a parameter of {name}")
        self._candidates, self._offset, self._apply = candidates, current_offset, apply
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{name}</b>  <span style='color:#8a9099'>follows another component's "
                             "parameter, plus a fixed offset</span>"))
        row = QHBoxLayout()
        self.target = QComboBox()
        for key, label in targets:
            self.target.addItem(label, key)
        self.source = QComboBox()
        self.offset = QLineEdit()
        self.offset.setFixedWidth(90)
        for w in (self.target, QLabel("="), self.source, QLabel("+"), self.offset):
            row.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(QLabel("<span style='color:#8a9099'>the offset starts at the current difference, so "
                             "nothing moves; 0 puts them together</span>"))
        self.error = QLabel("")
        self.error.setStyleSheet("color: #f06b6b;")
        self.error.setWordWrap(True)
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.target.currentIndexChanged.connect(self._fill_sources)
        self.source.currentIndexChanged.connect(self._fill_offset)
        self._fill_sources()

    def _fill_sources(self, *_):
        self.source.blockSignals(True)
        self.source.clear()
        key = self.target.currentData()
        if key is not None:
            self.source.addItems(self._candidates(key))
        self.source.blockSignals(False)
        none = self.source.count() == 0
        self.error.setText("no other component in this plane has this parameter (untied)" if none else "")
        self.ok_button.setEnabled(not none)
        self._fill_offset()

    def _fill_offset(self, *_):
        src = self.source.currentText()
        self.offset.setText(f"{self._offset(self.target.currentData(), src):.6g}" if src else "")

    def ok(self):
        """Try the tie; close only if it was accepted."""
        try:
            err = self._apply(self.target.currentData(), self.source.currentText(), parse_value(self.offset.text()))
        except ValueError as e:
            err = f"offset: {e}"
        if err:
            self.error.setText(err)
            return
        self.accept()
