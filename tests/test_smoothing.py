"""DS9-like display smoothing (forwardgal.smoothing)."""

import numpy as np
import pytest

from forwardgal.smoothing import Smooth, kernel, smooth


def test_kernels():
    b = kernel("boxcar", 2)
    assert b.shape == (5, 5) and np.allclose(b, 1 / 25)
    t = kernel("tophat", 2)
    assert t[0, 0] == 0 and t[2, 0] > 0 and t.sum() == pytest.approx(1)  # a disk: corners out
    assert (t > 0).sum() == 13  # 13 pixels within d <= 2
    g = kernel("gaussian", 3)
    assert g.sum() == pytest.approx(1) and g[3, 3] == g.max()
    assert g[3, 4] / g[3, 3] == pytest.approx(np.exp(-1 / (2 * 1.5**2)))  # sigma = r / 2
    g2 = kernel("gaussian", 3, sigma=0.5)
    assert g2[3, 4] / g2[3, 3] == pytest.approx(np.exp(-2.0))


def test_smooth_off_is_the_identity_and_flux_of_a_flat_image_is_kept():
    a = np.random.default_rng(0).normal(size=(20, 20))
    np.testing.assert_array_equal(smooth(a, Smooth()), a)
    flat = np.full((15, 15), 3.0)
    np.testing.assert_allclose(smooth(flat, Smooth(on=True, function="gaussian", radius=4)), 3.0)  # edges too


def test_boxcar_is_the_local_mean():
    a = np.arange(49.0).reshape(7, 7)
    out = smooth(a, Smooth(on=True, radius=1))
    assert out[3, 3] == pytest.approx(a[2:5, 2:5].mean())
    assert out[0, 0] == pytest.approx(a[0:2, 0:2].mean())  # at the edge: the mean of what is there


def test_normalized_over_the_pixels_with_values():
    """Outside a mask (NaN): not counted, and still NaN; the edge keeps its level (not darkened)."""
    a = np.full((9, 9), 5.0)
    a[:, :4] = np.nan
    out = smooth(a, Smooth(on=True, function="tophat", radius=2))
    assert np.all(np.isnan(out[:, :4])) and np.allclose(out[:, 4:], 5.0)


def test_bad_parameters():
    with pytest.raises(ValueError, match="function"):
        Smooth(function="median")
    with pytest.raises(ValueError, match="radius"):
        Smooth(radius=0)
    with pytest.raises(ValueError, match="sigma"):
        Smooth(sigma=-1.0)
