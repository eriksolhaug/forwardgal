"""Display scaling of images, DS9-like (D2, 2026-11-23). No Qt here.

An image value v is shown as a colour-map position y in [0, 1]:

1. limits [lo, hi] (the ``limits`` mode): "minmax", "zscale" (IRAF's
   algorithm, astropy.visualization.ZScaleInterval), a percentile range
   "99.5" / "99" / "98" / "95" / "90" (the central fraction of the pixels),
   "user" (Scale.lo, Scale.hi), or "default" (the panel's own default);
2. x = clip((v - lo) / (hi - lo), 0, 1);
3. the ``scale`` (DS9's definitions):
       linear   y = x
       log      y = log10(a x + 1) / log10(a + 1),   a = 1000 (DS9's default exponent; DS9
                divides by log10(a), which sends x = 1 to 1.00015; ours sends it to 1)
       sqrt     y = sqrt(x)
       squared  y = x^2
       asinh    y = asinh(10 x) / 3
       histeq   y = the fraction of the (sample) pixels in [lo, hi] below v
   or "default": the panel's default (forwardgal's look: e.g. asinh(v / s) with
   s ~ the noise, then linear between the default limits);
4. contrast c and bias b, as DS9 applies them to the colour map:
       y' = clip((y - b) c + 0.5, 0, 1),   defaults c = 1, b = 0.5 (no change).

``Default`` describes a panel's default: ``pre`` (None, or "asinh" with
``soft`` = s: v -> asinh(v / s)) and ``limits`` ("pct", p_lo, p_hi) /
("fixed", lo, hi) / ("zero_pct", p_hi) (lo = 0). Limits for "default" are
computed on ``default_sample`` (e.g. the data in the mask, so Model and
Data - model share the data's limits); every other mode on ``sample`` (the
shown array's own pixels in the mask).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SCALES = ("default", "linear", "log", "sqrt", "squared", "asinh", "histeq")
LIMITS = ("default", "minmax", "zscale", "99.5", "99", "98", "95", "90", "user")
LOG_A = 1000.0


@dataclass(frozen=True)
class Scale:
    """One display's scaling (the Scale menu); the defaults change nothing."""

    scale: str = "default"
    limits: str = "default"
    lo: float | None = None  # "user" limits, in data units
    hi: float | None = None
    contrast: float = 1.0
    bias: float = 0.5

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {self.scale!r}")
        if self.limits not in LIMITS:
            raise ValueError(f"limits must be one of {LIMITS}, got {self.limits!r}")
        if self.limits == "user" and (self.lo is None or self.hi is None or not self.lo < self.hi):
            raise ValueError(f"user limits need lo < hi, got ({self.lo}, {self.hi})")

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Default:
    """A panel's default scaling (see the module docstring)."""

    pre: str | None = None  # None or "asinh"
    soft: float = 1.0
    limits: tuple = ("pct", 0.5, 99.7)


def _pre(v, default):
    return np.arcsinh(np.asarray(v, float) / default.soft) if default.pre == "asinh" else np.asarray(v, float)


def limits(sample, mode, scale=None):
    """(lo, hi) of the finite values ``sample`` for a limits ``mode`` (not "default")."""
    s = np.asarray(sample, float)
    s = s[np.isfinite(s)]
    if mode == "user":
        return float(scale.lo), float(scale.hi)
    if s.size == 0:
        return 0.0, 1.0
    if mode == "minmax":
        lo, hi = float(s.min()), float(s.max())
    elif mode == "zscale":
        from astropy.visualization import ZScaleInterval

        lo, hi = (float(x) for x in ZScaleInterval().get_limits(s))
    else:  # a central percentage
        f = float(mode)
        lo, hi = (float(x) for x in np.percentile(s, [(100 - f) / 2, 100 - (100 - f) / 2]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def default_limits(sample, default):
    """(lo, hi) of a panel's default, on the pre-transformed ``sample``."""
    kind = default.limits[0]
    if kind == "fixed":
        return float(default.limits[1]), float(default.limits[2])
    s = _pre(sample, default)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0, 1.0
    if kind == "zero_pct":
        lo, hi = 0.0, float(np.percentile(s, default.limits[1]))
    else:
        lo, hi = (float(x) for x in np.percentile(s, [default.limits[1], default.limits[2]]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def stretch(x, scale):
    """y in [0, 1] from x in [0, 1] (DS9's scales; not "histeq" or "default")."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if scale == "linear":
        return x
    if scale == "log":
        return np.log10(LOG_A * x + 1.0) / np.log10(LOG_A + 1.0)
    if scale == "sqrt":
        return np.sqrt(x)
    if scale == "squared":
        return x * x
    if scale == "asinh":
        return np.arcsinh(10.0 * x) / 3.0
    raise ValueError(f"no stretch {scale!r}")


def colormap_position(y, contrast=1.0, bias=0.5):
    """DS9's contrast and bias: y' = clip((y - b) c + 0.5, 0, 1)."""
    return np.clip((np.asarray(y, float) - bias) * contrast + 0.5, 0.0, 1.0)


def apply(arr, scale, default, sample=None, default_sample=None):
    """Colour-map positions (same shape as ``arr``, in [0, 1]) and the (lo, hi) used.

    A NaN pixel (e.g. outside the mask of a residual) is shown as the value 0
    would be: white in the middle of the residual's diverging colour map.

    arr : the image shown; sample : its pixels for the limits (default: all of arr);
    default_sample : the pixels for "default" limits (default: sample).
    (lo, hi) are in the units the limits act on: pre-transformed for the default scale.
    """
    v = np.asarray(arr, float)
    sample = v if sample is None else np.asarray(sample, float)
    v = np.where(np.isfinite(v), v, 0.0)
    default_sample = sample if default_sample is None else default_sample
    use_pre = scale.scale == "default"
    if use_pre:
        v, sample_t = _pre(v, default), _pre(sample, default)
    else:
        sample_t = sample
    if scale.limits == "default":
        if use_pre:
            lo, hi = default_limits(default_sample, default)
        else:  # a DS9 scale with the default limits: the default's percentages, on the plain values
            lo, hi = default_limits(default_sample, replace(default, pre=None))
    elif scale.limits == "user" and use_pre:
        lo, hi = (float(x) for x in _pre(np.array([scale.lo, scale.hi]), default))
    else:
        lo, hi = limits(sample_t, scale.limits, scale)
    x = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    if scale.scale == "histeq":
        s = np.sort(sample_t[np.isfinite(sample_t) & (sample_t >= lo) & (sample_t <= hi)])
        y = np.searchsorted(s, v, side="right") / max(s.size, 1) if s.size else x
    elif use_pre:
        y = x
    else:
        y = stretch(x, scale.scale)
    out = colormap_position(y, scale.contrast, scale.bias)
    return out, (lo, hi)
