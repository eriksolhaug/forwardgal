"""D2: DS9-like display scaling (forwardgal.scaling)."""

import numpy as np
import pytest

from forwardgal.scaling import Default, Scale, apply, colormap_position, limits, stretch


@pytest.mark.parametrize("scale, x, y", [
    ("linear", 0.25, 0.25), ("sqrt", 0.25, 0.5), ("squared", 0.5, 0.25),
    ("log", 1.0, 1.0), ("log", 0.0, 0.0), ("log", 0.001, np.log10(2.0) / np.log10(1001.0)),
    ("asinh", 0.0, 0.0), ("asinh", 1.0, np.arcsinh(10.0) / 3.0),
])
def test_ds9_stretches(scale, x, y):
    assert float(stretch(x, scale)) == pytest.approx(y)


def test_stretches_are_monotonic_and_clipped():
    x = np.linspace(-0.5, 1.5, 201)
    for s in ("linear", "log", "sqrt", "squared", "asinh"):
        y = stretch(x, s)
        assert np.all(np.diff(y) >= 0) and y.min() >= 0 and y[x >= 1].max() == pytest.approx(stretch(1.0, s))


def test_contrast_and_bias():
    y = np.linspace(0, 1, 11)
    np.testing.assert_allclose(colormap_position(y), y)  # defaults: no change
    np.testing.assert_allclose(colormap_position(y, contrast=2.0), np.clip(2 * y - 0.5, 0, 1))
    np.testing.assert_allclose(colormap_position(y, bias=0.3), np.clip(y + 0.2, 0, 1))  # lower bias: brighter


def test_limits_modes():
    s = np.arange(1001.0)  # 0 .. 1000
    assert limits(s, "minmax") == (0.0, 1000.0)
    assert limits(s, "99") == pytest.approx((5.0, 995.0))
    assert limits(s, "90") == pytest.approx((50.0, 950.0))
    lo, hi = limits(np.r_[np.random.default_rng(1).normal(0, 1, 5000), 1e6], "zscale")
    assert -5 < lo < 0 < hi < 5  # zscale ignores the outlier
    assert limits(s, "user", Scale(limits="user", lo=3.0, hi=7.0)) == (3.0, 7.0)
    with pytest.raises(ValueError, match="lo < hi"):
        Scale(limits="user", lo=7.0, hi=3.0)
    with pytest.raises(ValueError, match="scale must be"):
        Scale(scale="loggy")


def test_default_reproduces_the_old_look():
    """Default (asinh(v / soft), data percentiles) = what the GUI drew before D3."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.1, (40, 40)) + 5 * np.exp(-((np.indices((40, 40)) - 20) ** 2).sum(0) / 20)
    soft = 0.1
    d = Default(pre="asinh", soft=soft, limits=("pct", 0.5, 99.7))
    y, (lo, hi) = apply(data, Scale(), d)
    t = np.arcsinh(data / soft)
    lo0, hi0 = np.percentile(t, [0.5, 99.7])
    assert (lo, hi) == pytest.approx((lo0, hi0))
    np.testing.assert_allclose(y, np.clip((t - lo0) / (hi0 - lo0), 0, 1))
    # residual default: linear, fixed -5..5
    r = rng.normal(0, 2, (10, 10))
    y, lim = apply(r, Scale(), Default(limits=("fixed", -5.0, 5.0)))
    assert lim == (-5.0, 5.0)
    np.testing.assert_allclose(y, np.clip((r + 5) / 10, 0, 1))


def test_model_shares_the_data_default_limits_but_not_others():
    data = np.linspace(0, 10, 100)
    model = data / 2
    d = Default(limits=("pct", 0.0, 100.0))
    _, lim = apply(model, Scale(), d, default_sample=data)
    assert lim == pytest.approx((0.0, 10.0))  # the data's
    _, lim = apply(model, Scale(scale="linear", limits="minmax"), d, default_sample=data)
    assert lim == pytest.approx((0.0, 5.0))  # its own


def test_histeq_flattens_and_user_limits_with_the_default_scale():
    v = np.r_[np.zeros(90), np.linspace(1, 100, 10)]
    y, _ = apply(v, Scale(scale="histeq", limits="minmax"), Default())
    assert y.max() == pytest.approx(1.0) and y[0] == pytest.approx(0.9)  # 90 % of pixels at or below 0
    d = Default(pre="asinh", soft=2.0)
    y, (lo, hi) = apply(np.array([0.0, 10.0]), Scale(limits="user", lo=0.0, hi=10.0), d)
    assert (lo, hi) == pytest.approx((0.0, np.arcsinh(5.0))) and y.tolist() == pytest.approx([0.0, 1.0])


def test_nan_shows_as_the_value_zero():
    y, _ = apply(np.array([np.nan, 1.0, 2.0]), Scale(scale="linear", limits="minmax"), Default(),
                 sample=np.array([1.0, 2.0]))
    assert y.tolist() == [0.0, 0.0, 1.0]  # 0 is below the limits
    y, _ = apply(np.array([np.nan, -5.0, 5.0]), Scale(), Default(limits=("fixed", -5.0, 5.0)))
    assert y.tolist() == [0.5, 0.0, 1.0]  # a residual: NaN in the middle (white), not at -5 sigma
