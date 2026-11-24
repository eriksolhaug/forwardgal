"""Display smoothing, DS9-like (2026-11-23). No Qt here; the fit never sees it.

Kernels on a (2r + 1) x (2r + 1) grid of pixel offsets (dx, dy), d^2 = dx^2 + dy^2:

    boxcar    K = 1                              (a square, as DS9)
    tophat    K = 1 where d <= r, else 0         (a disk)
    gaussian  K = exp(-d^2 / (2 sigma^2)), d <= r   (sigma: default r / 2, as DS9)

each normalized to sum 1. Smoothing is NORMALIZED (user, 2026-11-23): with
w = 1 where a pixel has a value (finite; e.g. inside the mask for the
residual) and 0 elsewhere,

    out = conv(K, w v) / conv(K, w)

so the empty outside of a mask does not darken its edge; pixels with no
value stay without one (NaN).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FUNCTIONS = ("boxcar", "tophat", "gaussian")


@dataclass(frozen=True)
class Smooth:
    """One display's smoothing (the Smooth menu); off by default (DS9's defaults otherwise)."""

    on: bool = False
    function: str = "boxcar"
    radius: int = 3  # pixels
    sigma: float | None = None  # gaussian; None: radius / 2

    def __post_init__(self):
        if self.function not in FUNCTIONS:
            raise ValueError(f"function must be one of {FUNCTIONS}, got {self.function!r}")
        if not (isinstance(self.radius, (int, np.integer)) and self.radius >= 1):
            raise ValueError(f"radius must be a whole number of pixels >= 1, got {self.radius!r}")
        if self.sigma is not None and not self.sigma > 0:
            raise ValueError(f"sigma must be positive, got {self.sigma!r}")

    def with_(self, **kw):
        return replace(self, **kw)


def kernel(function, radius, sigma=None):
    """Normalized (2r + 1) x (2r + 1) kernel, indexed [row = dy + r, col = dx + r]."""
    r = int(radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = dx * dx + dy * dy
    if function == "boxcar":
        k = np.ones(d2.shape)
    elif function == "tophat":
        k = (d2 <= r * r).astype(float)
    elif function == "gaussian":
        s = r / 2.0 if sigma is None else float(sigma)
        k = np.where(d2 <= r * r, np.exp(-d2 / (2.0 * s * s)), 0.0)
    else:
        raise ValueError(f"function must be one of {FUNCTIONS}, got {function!r}")
    return k / k.sum()


def smooth(arr, s):
    """The array smoothed by ``s`` (a Smooth), normalized over the pixels with values; ``arr`` itself if off."""
    if not s.on:
        return np.asarray(arr, float)
    from scipy.ndimage import convolve

    v = np.asarray(arr, float)
    w = np.isfinite(v).astype(float)
    k = kernel(s.function, s.radius, s.sigma)
    num = convolve(np.where(w > 0, v, 0.0), k, mode="constant", cval=0.0)
    den = convolve(w, k, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(w > 0, out, np.nan)
