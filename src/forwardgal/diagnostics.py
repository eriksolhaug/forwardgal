"""Checks that are run on finished fits and saved with them."""

from __future__ import annotations

import warnings

import numpy as np


def rendering_check(
    model_hybrid,
    model_exact,
    data,
    sigma,
    mask,
    convolve=None,
    max_dev_limit=0.1,
    dchi2_limit=1.0,
):
    """Compare the fast (hybrid) rendering of a model with the exact one.

    Both models are first passed through ``convolve`` (the PSF convolution;
    identity if None), because the data are compared with PSF-convolved
    models. Then, inside the mask,

        max_dev = max |m_hybrid - m_exact| / sigma
        dchi2   = chi2_hybrid - chi2_exact,  chi2 = sum ((data - m) / sigma)^2

    Parameters
    ----------
    model_hybrid, model_exact : (ny, nx) arrays
        Unconvolved model images (same units as data).
    data, sigma : (ny, nx) arrays
    mask : (ny, nx) bool array
        True for pixels used in the fit.
    convolve : callable or None
        Maps an image to the PSF-convolved image.
    max_dev_limit, dchi2_limit : float
        A warning is issued if max_dev > max_dev_limit or |dchi2| > dchi2_limit.

    Returns
    -------
    dict with keys "max_dev_sigma", "delta_chi2", "ok" (bool).
    """
    if convolve is None:
        convolve = lambda image: image  # noqa: E731
    m_h = np.asarray(convolve(model_hybrid))[mask]
    m_e = np.asarray(convolve(model_exact))[mask]
    d, s = np.asarray(data)[mask], np.asarray(sigma)[mask]

    max_dev = float(np.max(np.abs(m_h - m_e) / s))
    dchi2 = float(np.sum(((d - m_h) / s) ** 2) - np.sum(((d - m_e) / s) ** 2))
    ok = (max_dev <= max_dev_limit) and (abs(dchi2) <= dchi2_limit)
    if not ok:
        warnings.warn(
            f"rendering check failed: max |hybrid - exact| / sigma = {max_dev:.3g} "
            f"(limit {max_dev_limit}), delta chi2 = {dchi2:.3g} (limit {dchi2_limit}). "
            "Increase RenderSettings.subsample or .patch for this fit.",
            stacklevel=2,
        )
    return {"max_dev_sigma": max_dev, "delta_chi2": dchi2, "ok": ok}
