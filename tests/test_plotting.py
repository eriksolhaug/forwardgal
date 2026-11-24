import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from forwardgal.plotting import corner_mask, grid_lines, plot_traced_grid  # noqa: E402


def test_corner_mask():
    m = np.zeros((3, 4), bool)
    m[1, 2] = True
    c = corner_mask(m)
    assert c.shape == (4, 5)
    assert c.sum() == 4 and c[1, 2] and c[1, 3] and c[2, 2] and c[2, 3]


def test_grid_lines_every_and_gaps():
    cx, cy = np.meshgrid(np.arange(7.0), np.arange(5.0))
    keep = np.ones_like(cx, bool)
    keep[2, 3] = False
    rows, cols = grid_lines(cx, cy, keep, every=2)
    assert len(rows) == 3 and len(cols) == 4  # rows 0, 2, 4; columns 0, 2, 4, 6
    assert np.isnan(rows[1][0][3])  # the dropped corner breaks row 2


def test_plot_traced_grid_runs(tmp_path):
    from test_render_lensed import grid_and_centre

    grid, *_ = grid_and_centre(0.45, 0.45)
    mask = np.zeros(grid.shape, bool)
    mask[10:30, 15:25] = True
    fig, ax = plt.subplots()
    plot_traced_grid(ax, grid, mask, every=4)
    plot_traced_grid(ax, grid, mask, every=4, plane="image")
    assert len(ax.lines) > 0
    fig.savefig(tmp_path / "g.png")
