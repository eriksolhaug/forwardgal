"""Elliptical Sersic profile (GALFIT conventions).

    I(R) = I_e * exp(-b_n * [(R / R_e)^(1/n) - 1])

Geometry
--------
Positions (x, y) are (theta_x, theta_y) or (beta_x, beta_y) in arcsec, with
x = +east and y = +north. PA is the angle of the MAJOR axis measured east of
north (GALFIT): PA = 0 points north (+y), PA = 90 deg points east (+x).
With (dx, dy) = (x - x0, y - y0):

    u = dx sin(PA) + dy cos(PA)      (along the major axis)
    v = dx cos(PA) - dy sin(PA)      (along the minor axis)
    R = sqrt(u^2 + (v / q)^2)

so R_e is the effective radius along the major axis and q = b/a <= 1.

Units
-----
I_e has the units of the image (e.g. MJy/sr). R_e, x0, y0 are in arcsec.

Point values vs pixel values
----------------------------
``sersic`` returns the surface brightness at points. A model pixel must be the
average over the pixel area. Uniform subsampling approximates that well where
the profile is smooth, but fails near the centre of steep profiles (n >= 4:
errors of tens of percent to factors of 100+, depending on where the centre
falls). Pixels near the centre are therefore integrated exactly with the polar
method (see forwardgal.geometry and the README, "How pixels are rendered").
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln


def b_n(n):
    """Sersic b_n such that half of the total light lies within R_e.

    b_n solves P(2n, b_n) = 1/2, with P the regularized lower incomplete gamma
    function. Start from the Ciotti & Bertin (1999) series (relative error
    ~4e-4 at n = 0.5, ~1e-6 at n = 2) and refine with two Newton steps,
    using dP/db = b^(2n-1) e^(-b) / Gamma(2n). Result is exact to ~1e-12 for
    0.3 <= n <= 10, and differentiable in JAX.
    """
    b = (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )
    a = 2.0 * n
    for _ in range(2):
        residual = gammainc(a, b) - 0.5
        slope = jnp.exp((a - 1.0) * jnp.log(b) - b - gammaln(a))
        b = b - residual / slope
    return b


def elliptical_radius(x, y, x0, y0, q, pa_deg):
    """Elliptical radius R = sqrt(u^2 + (v/q)^2) in arcsec (see module docstring)."""
    pa = jnp.deg2rad(pa_deg)
    dx, dy = x - x0, y - y0
    u = dx * jnp.sin(pa) + dy * jnp.cos(pa)
    v = dx * jnp.cos(pa) - dy * jnp.sin(pa)
    return jnp.sqrt(u**2 + (v / q) ** 2)


def sersic(x, y, x0, y0, I_e, R_e, n, q, pa_deg):
    """Sersic surface brightness at positions (x, y).

    Parameters
    ----------
    x, y : arrays (arcsec)
        Evaluation positions, +x east, +y north. Any matching shapes.
    x0, y0 : float (arcsec)
        Centre.
    I_e : float
        Surface brightness at R = R_e (image units, e.g. MJy/sr).
    R_e : float (arcsec)
        Effective (half-light) radius along the major axis.
    n : float
        Sersic index.
    q : float
        Axis ratio b/a, 0 < q <= 1.
    pa_deg : float
        Major-axis position angle in degrees, east of north.

    Returns
    -------
    array with the shape of x: surface brightness in the units of I_e.
    """
    R = elliptical_radius(x, y, x0, y0, q, pa_deg)
    return I_e * jnp.exp(-b_n(n) * ((R / R_e) ** (1.0 / n) - 1.0))


def sersic_total_flux(I_e, R_e, n, q):
    """Total flux of an elliptical Sersic profile, integrated to infinity.

        L = 2 pi n q R_e^2 I_e exp(b_n) b_n^(-2n) Gamma(2n)

    Parameters
    ----------
    I_e : float
        Surface brightness at R_e (image units, e.g. MJy/sr).
    R_e : float
        Effective radius along the major axis (arcsec).
    n : float
        Sersic index.
    q : float
        Axis ratio b/a.

    Returns
    -------
    float
        L in (image units) x arcsec^2, e.g. MJy/sr * arcsec^2.
    """
    # TODO(Erik): implement the formula above with jax.numpy (jnp).
    #   - Use b_n(n) from this module.
    #   - For Gamma(2n) use jax.scipy.special.gamma, or better
    #     jnp.exp(jax.scipy.special.gammaln(2 * n)) -- think about why gammaln
    #     might be preferred (hint: what happens to Gamma(2n) for large n?).
    # Check your version with:  pytest tests/test_sersic.py
    L = 2 * jnp.pi * n * q * R_e**2 * I_e * jnp.exp(b_n(n)) * b_n(n)**(-2*n) * jnp.exp(gammaln(2*n))
    return L


def mag_to_ie(m_tot, R_e, n, q, zeropoint, pixscale):
    """Convert total magnitude to I_e.

    The sum of pixel values of a profile with total flux L is L / A_pix, with
    A_pix = pixscale^2 the pixel area in arcsec^2, and
    m_tot = -2.5 log10(L / A_pix) + zeropoint. Since L is proportional to I_e,

        I_e = 10^(-0.4 (m_tot - zeropoint)) * A_pix / L(I_e = 1).
    """
    pixel_sum = 10.0 ** (-0.4 * (m_tot - zeropoint))
    return pixel_sum * pixscale**2 / sersic_total_flux(1.0, R_e, n, q)


def ie_to_mag(I_e, R_e, n, q, zeropoint, pixscale):
    """Convert I_e to total magnitude (inverse of mag_to_ie)."""
    pixel_sum = sersic_total_flux(I_e, R_e, n, q) / pixscale**2
    return -2.5 * jnp.log10(pixel_sum) + zeropoint
