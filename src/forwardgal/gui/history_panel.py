"""The model-history card at the bottom of the window (G3b), after the mockup.

One row per fit or draft folder (forwardgal.history.list_history):

    Model | Components | Started from | Fit | χ²/ν | BIC | Saved | Load  🗑

The loaded model's row is highlighted and says "Loaded" instead of a Load
button. The header has "Save to history" (``save_clicked()``) and
"Trash (n)" (a menu of the trashed folders; choosing one emits
``restore_clicked(name)``). Row buttons emit ``load_clicked(folder)`` and
``trash_clicked(folder)``; a fit's row also has "Use in band…"
(``use_in_band_clicked(folder)``: seed another band from it, G5). The panel only shows rows: the window does the
work and calls ``set_rows`` again.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MUTED = "#8a9099"
COLUMNS = ("Model", "Components", "Started from", "Fit", "χ²/ν", "BIC", "Saved", "")


def _fmt(x, f):
    return "—" if x is None else format(x, f)


def _saved(created):
    """'2026-11-23T14:02:11' -> 'Nov 23 14:02' (as in the mockup)."""
    import datetime as dt

    try:
        return dt.datetime.fromisoformat(str(created)).strftime("%b %d %H:%M")
    except ValueError:
        return str(created)


class HistoryPanel(QWidget):
    """Model history table + "Save to history" and "Trash (n)"."""

    load_clicked = Signal(str)  # folder name
    trash_clicked = Signal(str)  # folder name
    restore_clicked = Signal(str)  # name in .trash
    save_clicked = Signal()
    use_in_band_clicked = Signal(str)  # folder name (a fit)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        self.title = QLabel("<b>Model history</b>")
        head.addWidget(self.title)
        head.addStretch(1)
        self.save_button = QPushButton("Save to history")
        self.save_button.setToolTip("save the current model as a new numbered draft (not fitted)")
        self.save_button.clicked.connect(self.save_clicked.emit)
        head.addWidget(self.save_button)
        self.trash_button = QPushButton("Trash (0)")
        self.trash_button.setToolTip("trashed fits and drafts (in .trash/: choose one to restore it)")
        self.trash_menu = QMenu(self)
        self.trash_menu.triggered.connect(lambda a: self.restore_clicked.emit(a.data()))
        self.trash_button.setMenu(self.trash_menu)
        head.addWidget(self.trash_button)
        lay.addLayout(head)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)  # one line per row: rows sized once while the table is still narrow
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(30)  # would otherwise wrap and stay tall
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(self.table)
        self.rows = []
        self.loaded = None

    def set_rows(self, rows, loaded=None, trashed=(), title=None):
        """Show ``rows`` (history.list_history); ``loaded``: the folder now in the GUI (or None)."""
        self.rows, self.loaded = list(rows), loaded
        if title is not None:
            self.title.setText(f"<b>Model history</b> <span style='color:{MUTED}'>· {title}</span>")
        self.table.setRowCount(0)  # drop the old rows AND their button widgets (else stale ones linger)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            current = r["folder"] == loaded
            cells = [r["folder"] + ("  · current" if current else ""), r["components"], r["started_from"] or "—",
                     r["status"], _fmt(r["chi2_per_dof"], ".3f"), _fmt(r["bic"], ",.0f"), _saved(r["created"])]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if current:
                    item.setForeground(QColor("#f0a04b"))
                    item.setBackground(QColor("#2a2418"))
                if j in (4, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)
            self.table.setCellWidget(i, len(COLUMNS) - 1, self._actions(r["folder"], current, r["kind"] == "fit"))
        self.trash_menu.clear()
        for name in trashed:
            a = self.trash_menu.addAction(f"restore {name}")
            a.setData(name)
        self.trash_button.setText(f"Trash ({len(trashed)})")
        self.trash_button.setEnabled(bool(trashed))

    def _actions(self, folder, current, is_fit=False):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 0, 2, 0)
        load = QPushButton("Loaded" if current else "Load")
        load.setEnabled(not current)
        load.setToolTip("put this model in the GUI (a fit: best-fit or starting values)")
        load.clicked.connect(lambda: self.load_clicked.emit(folder))
        bin_ = QPushButton("🗑")
        bin_.setToolTip("move to the trash (history/.trash: recoverable)")
        bin_.clicked.connect(lambda: self.trash_clicked.emit(folder))
        h.addWidget(load)
        use = QPushButton("Use in band…")
        use.setToolTip("seed another band from this fit (its shapes within ranges; fluxes solved afresh)")
        use.clicked.connect(lambda: self.use_in_band_clicked.emit(folder))
        use.setVisible(is_fit)
        h.addWidget(use)
        h.addWidget(bin_)
        return w

    def button(self, folder, which):
        """The "load" or "trash" button of ``folder``'s row (for tests)."""
        i = [r["folder"] for r in self.rows].index(folder)
        buttons = self.table.cellWidget(i, len(COLUMNS) - 1).findChildren(QPushButton)
        return buttons[{"load": 0, "use": 1, "trash": 2}[which]]
