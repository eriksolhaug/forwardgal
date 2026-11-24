import numpy as np
import pytest

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.regions import fit_region, polygon_mask
from forwardgal.render import render_sersic


def test_polygon_mask_square():
    # Square from row 2..5, col 3..7 (pixel centres strictly inside).
    m = polygon_mask((10, 10), [(1.5, 2.5), (1.5, 7.5), (5.5, 7.5), (5.5, 2.5)])
    expected = np.zeros((10, 10), bool)
    expected[2:6, 3:8] = True
    np.testing.assert_array_equal(m, expected)


def test_single_pixel_margin_is_a_disk():
    mask = np.zeros((21, 21), bool)
    mask[10, 10] = True
    reg = fit_region(mask, margin=3)
    # Pixels with integer offsets (i, j), i^2 + j^2 <= 9: Gauss circle count 29.
    assert reg.render.sum() == 29
    assert reg.box == (7, 14, 7, 14) and reg.box_shape == (7, 7)
    assert all(v == 3 for v in reg.margin_available.values())
    assert "margin complete" in reg.describe()


def test_render_includes_mask():
    rng = np.random.default_rng(0)
    mask = rng.uniform(size=(30, 30)) > 0.97
    reg = fit_region(mask, margin=2)
    assert np.all(reg.render[mask])


def test_truncated_margin_reported():
    mask = np.zeros((20, 20), bool)
    mask[1, 10] = True  # 1 px from the south edge (row 0), 9 px from east (col 0)
    reg = fit_region(mask, margin=5)
    assert reg.margin_available["south"] == 1.0
    assert reg.margin_available["north"] == 5.0
    text = reg.describe()
    assert "WARNING" in text and "south 1 px" in text


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        fit_region(np.zeros((5, 5), bool))


# --- rendering only the render pixels --------------------------------------------

KW = dict(pixscale=0.03, x0=0.004, y0=-0.008, I_e=1.0, R_e=0.09, n=2.0, q=0.6, pa_deg=30.0)


def test_render_all_pixels_equals_full_render():
    shape = (31, 33)
    rows, cols = np.nonzero(np.ones(shape, bool))
    full = np.asarray(render_sersic(shape, row0=15, col0=16, **KW))
    sub = np.asarray(render_sersic(shape, row0=15, col0=16, pixels=(rows, cols), **KW))
    np.testing.assert_allclose(sub, full, rtol=1e-13)


def test_render_subset_is_zero_elsewhere_except_patch():
    shape = (31, 33)
    mask = np.zeros(shape, bool)
    mask[:, :8] = True  # a strip far from the centre (col 16)
    rows, cols = np.nonzero(mask)
    full = np.asarray(render_sersic(shape, row0=15, col0=16, **KW))
    sub = np.asarray(render_sersic(shape, row0=15, col0=16, pixels=(rows, cols), **KW))
    np.testing.assert_allclose(sub[mask], full[mask], rtol=1e-13)
    patch = np.zeros(shape, bool)
    patch[12:19, 13:20] = True  # 7 x 7 around the centre pixel (15, 16)
    np.testing.assert_allclose(sub[patch], full[patch], rtol=1e-13)  # always computed
    assert np.all(sub[~mask & ~patch] == 0.0)


def test_ellipse_polygon_orientation_and_area():
    from forwardgal.regions import ellipse_polygon
    # PA = 0: major axis north (+row); PA = 90: east (-col).
    v = ellipse_polygon((50, 50), 20, 5, 0.0)
    assert np.ptp(v[:, 0]) == pytest.approx(40, abs=0.1) and np.ptp(v[:, 1]) == pytest.approx(10, abs=0.1)
    v = ellipse_polygon((50, 50), 20, 5, 90.0)
    assert np.ptp(v[:, 1]) == pytest.approx(40, abs=0.1)
    m = polygon_mask((101, 101), ellipse_polygon((50, 50), 30, 10, 37.0, n=360))
    assert m.sum() == pytest.approx(np.pi * 30 * 10, rel=0.02)
