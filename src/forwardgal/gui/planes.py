"""A plane panel (G1b): a pyqtgraph plot in arcsec with east to the LEFT
(inverted x axis), locked aspect ratio, one image, line/marker overlays and a
``moved(x, y)`` signal with the mouse position in data coordinates (arcsec).
Used for both the source plane (beta) and the image plane (theta).

Draw mode (A3b): while ``drawing`` is True, a left-button drag on the plot
draws a dashed preview ellipse (major axis from press to cursor; see
geometry.drag_to_ellipse) instead of panning, and on release emits
``drawn(x0, y0, x1, y1)``: the drag's start and end in arcsec. Other buttons
(and all drags when not drawing) behave as before.

Colorbar (D4): a thin bar under the plot shows the panel's colour map as it
is applied (after contrast and bias). Dragging on it changes them as in DS9
(right-drag on the image there): left / right moves the bias, up / down the
contrast; double-click or "Default" resets them. It emits
``stretch_changed(contrast, bias)``; the window owns the values (Scale).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

# Arrays are [row, col] everywhere in forwardgal: tell pyqtgraph.
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#0d0f12", foreground="#c9cdd3")

TEAL = "#3fb8a8"  # caustics and critical curves (as in the mockup)
GOLD = "#f2c14e"  # the quasar
COMPONENT_COLOURS = ["#f0a04b", "#5b9cf0", "#b58cf0", "#6cd28a", "#f06b8a", "#e8e06a"]


def asinh_stretch(x, soft):
    """arcsinh(x / soft): linear below ~soft (e.g. the noise), logarithmic above."""
    return np.arcsinh(np.asarray(x, float) / soft)


class ColorBar(QWidget):
    """The colour map as shown (after contrast c and bias b); drag to change c and b (DS9-like)."""

    changed = Signal(float, float)  # contrast, bias
    BIAS_PER_WIDTH = 1.0  # dragging across the whole bar moves the bias by this
    CONTRAST_E_FOLD_PX = 60.0  # dragging up this many pixels multiplies the contrast by e

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.setCursor(Qt.SizeAllCursor)
        self.setToolTip("drag: left / right = bias, up / down = contrast (as in DS9); double-click: default")
        self.cmap = pg.colormap.get("CET-L1")
        self.contrast, self.bias = 1.0, 0.5
        self._start = None

    def set_state(self, cmap, contrast, bias):
        self.cmap, self.contrast, self.bias = cmap, float(contrast), float(bias)
        self.update()

    def paintEvent(self, _event):  # noqa: N802 (Qt's name)
        from forwardgal.scaling import colormap_position

        w = max(self.width(), 1)
        y = colormap_position(np.linspace(0.0, 1.0, w), self.contrast, self.bias)
        colours = self.cmap.map(y, mode="qcolor")
        p = QPainter(self)
        for x, c in enumerate(colours):
            p.setPen(QColor(c))
            p.drawLine(x, 0, x, self.height())
        p.end()

    def mousePressEvent(self, ev):  # noqa: N802
        self._start = (ev.position(), self.contrast, self.bias)

    def mouseMoveEvent(self, ev):  # noqa: N802
        if self._start is None:
            return
        p0, c0, b0 = self._start
        d = ev.position() - p0
        self.drag_to(c0, b0, d.x(), d.y())

    def mouseReleaseEvent(self, _ev):  # noqa: N802
        self._start = None

    def mouseDoubleClickEvent(self, _ev):  # noqa: N802
        self.changed.emit(1.0, 0.5)

    def drag_to(self, c0, b0, dx, dy):
        """The contrast and bias after dragging (dx, dy) pixels from (c0, b0); emits ``changed``."""
        bias = float(np.clip(b0 - self.BIAS_PER_WIDTH * dx / max(self.width(), 1), -1.0, 2.0))
        contrast = float(np.clip(c0 * np.exp(-dy / self.CONTRAST_E_FOLD_PX), 0.05, 20.0))  # up = more
        self.contrast, self.bias = contrast, bias
        self.update()
        self.changed.emit(contrast, bias)


class PlanePanel(QWidget):
    """Title row (with room for buttons) above a pyqtgraph plot."""

    moved = Signal(float, float)
    stretch_changed = Signal(float, float)  # the colorbar dragged: contrast, bias
    drawn = Signal(float, float, float, float)  # a finished draw-mode drag: start (x, y), end (x, y)

    def __init__(self, title, subtitle, xlabel, ylabel, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        name = QLabel(f"<b>{title}</b><br><span style='color:#8a9099'>{subtitle}</span>")
        head.addWidget(name)
        head.addStretch(1)
        self.buttons = head  # callers may add buttons here
        lay.addLayout(head)
        self.plot = pg.PlotWidget()
        self.view = self.plot.getViewBox()
        self.view.invertX(True)  # east (+x) on the left, as on the sky
        self.view.setAspectLocked(True)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        lay.addWidget(self.plot, 1)
        row = QHBoxLayout()
        self.colorbar = ColorBar()
        self.colorbar.changed.connect(self.stretch_changed.emit)
        self.colorbar_reset = QPushButton("Default")
        self.colorbar_reset.setToolTip("contrast and bias back to their defaults")
        self.colorbar_reset.setFixedHeight(18)
        self.colorbar_reset.setStyleSheet("font-size: 10px; padding: 0px 6px;")  # fits the bar's height
        self.colorbar_reset.clicked.connect(lambda: self.stretch_changed.emit(1.0, 0.5))
        row.addWidget(self.colorbar, 1)
        row.addWidget(self.colorbar_reset)
        lay.addLayout(row)
        self._overlays = []
        self.plot.scene().sigMouseMoved.connect(self._on_mouse)
        self.drawing = False
        self.preview = pg.PlotDataItem(pen=pg.mkPen("w", width=1.5, style=Qt.DashLine))
        self.preview.setZValue(30)
        self.preview.hide()
        self.plot.addItem(self.preview)
        # pyqtgraph calls the ViewBox's mouseDragEvent for a drag on empty plot area: take it over in
        # draw mode, and hand everything else to the original (pan, zoom box).
        self._view_drag = self.view.mouseDragEvent
        self.view.mouseDragEvent = self._drag

    def set_image(self, arr, transform, levels, cmap):
        """Show a 2-D array (row-major) placed by a QTransform (item -> data coords)."""
        self.image.setImage(np.nan_to_num(np.asarray(arr, float)), levels=levels, autoLevels=False)
        self.image.setTransform(transform)
        self.image.setColorMap(cmap)

    def add_curve(self, xy, pen):
        item = pg.PlotDataItem(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1], pen=pen)
        self.plot.addItem(item)
        self._overlays.append(item)
        return item

    def add_marker(self, x, y, colour, symbol="+", size=16):
        item = pg.ScatterPlotItem([x], [y], symbol=symbol, size=size, pen=pg.mkPen(colour, width=2), brush=None)
        self.plot.addItem(item)
        self._overlays.append(item)
        return item

    def clear_overlays(self):
        for item in self._overlays:
            self.plot.removeItem(item)
        self._overlays = []

    def show_range(self, xlim, ylim):
        self.view.setRange(xRange=sorted(xlim), yRange=sorted(ylim), padding=0.02)

    def _drag(self, ev, axis=None):
        if not self.drawing or ev.button() != Qt.MouseButton.LeftButton:
            return self._view_drag(ev, axis)
        from forwardgal.gui.geometry import drag_to_ellipse  # (no import cycle: geometry has no Qt)

        ev.accept()
        a = self.view.mapSceneToView(ev.buttonDownScenePos(Qt.MouseButton.LeftButton))
        b = self.view.mapSceneToView(ev.scenePos())
        if ev.isFinish():
            self.preview.hide()
            self.drawn.emit(float(a.x()), float(a.y()), float(b.x()), float(b.y()))
            return
        x0, y0, R_e, q, pa = drag_to_ellipse((a.x(), a.y()), (b.x(), b.y()))
        t = np.linspace(0, 2 * np.pi, 97)
        u, v = R_e * np.cos(t), q * R_e * np.sin(t)  # along the major and minor axes
        s, c = np.sin(np.radians(pa)), np.cos(np.radians(pa))  # major axis (sin PA, cos PA); minor (cos, -sin)
        self.preview.setData(x0 + u * s + v * c, y0 + u * c - v * s)
        self.preview.show()

    def _on_mouse(self, pos):
        if self.plot.sceneBoundingRect().contains(pos):
            p = self.view.mapSceneToView(pos)
            self.moved.emit(float(p.x()), float(p.y()))
