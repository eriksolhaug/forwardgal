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

Fourier modes (GALFIT, Peng et al. 2010)
----------------------------------------
Optional azimuthal modes make the isophotes non-elliptical by scaling the
radius with an angle-dependent factor:

    R_F = R * g(theta_G),   g = 1 + sum_m a_m cos(m (theta_G + phi_m)),

with a_m the mode amplitude (dimensionless) and phi_m its phase (degrees,
GALFIT's numbers). theta_G is GALFIT's azimuthal angle in the stretched
(circularized) frame:

    theta_G = 180 deg - atan2(v / q, u).

This convention was MEASURED against GALFIT 3.0.5 model images (2026-11-16,
scripts/m12a_galfit_fourier.py): with it, GALFIT's images are reproduced to
~1e-4 for m = 1, 3, 4 at several PA, q, n. Consequence: for m = 1 with
phi_1 = 0 and a_1 > 0, g < 1 along +PA, so the light extends further toward
PA (the isophote radius is R_e / g).
g must stay > 0 at every angle (guaranteed if sum |a_m| < 1).

Bending modes (GALFIT "Bn"; M12f)
---------------------------------
The minor-axis coordinate is bent by a polynomial in the major-axis one:
with s = sign cos(PA - pa_ref) and the frame (u_B, v_B) = s (u, v), the
profile is evaluated at v_B' = v_B - sum_m a_m (-u_B / R_e)^m (u unchanged),
so the light ridge lies at v_B = sum_m a_m (-u_B / R_e)^m. a_m in arcsec
(the ridge shift at |u| = R_e). pa_ref places the seam where the frame
turns by 180 deg (PA = pa_ref +- 90): GALFIT uses pa_ref = 0; forwardgal
components use their starting PA (user, 2026-11-16). GALFIT's amplitude is
a_m (s s_G)^(m + 1) [arcsec; / pixel scale for GALFIT's pixels], s_G = s at
pa_ref = 0. MEASURED against GALFIT 3.0.5 (2026-11-16): 4e-7 of the peak,
with and without Fourier modes (measured in the bent frame). The map is a
shear (Jacobian 1): the total flux is unchanged.

Spiral rotation (GALFIT R0-R4, face-on) -- PAUSED (user, 2026-11-16)
---------------------------------------------------------------------
The code is commented out ("# [M12e paused]"); the measured GALFIT
definition is kept here and in MILESTONES.md for when it is resumed.
The PA turns with the circular radius r = |(dx, dy)| from the centre:
PA_eff(r) = PA + theta(r), with

    theta(r) = theta_out * 0.5 [tanh(x) + 1] * h(r),
    x = (r - r_in) / (r_out - r_in) * (2 - B) + B,
    B = 0.5 ln(0.46 / |theta_out [rad]|),
    h = [0.5 (r / r_out + 1)]^alpha              ("power"),
    h = ln(r / r_ws + 1) / ln(r_out / r_ws + 1)   ("log").

theta(r_out) = 0.982 theta_out (x = 2). Fourier modes are measured in the
rotated frame. MEASURED against GALFIT 3.0.5 (2026-11-16): agreement 1-6e-4
of the peak. An elongated profile (q < 1) under rotation gives two arms; a
mode m gives m arms. Inclination (R9) and sky PA (R10) are not implemented
(their GALFIT geometry is recorded in MILESTONES.md).

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

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammainc, gammaincc, gammaln

from forwardgal.geometry import angular_segments, ray_polygon_range


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


def major_minor(x, y, x0, y0, pa_deg):
    """(u, v): offsets along the major axis (toward PA) and the minor axis (arcsec)."""
    pa = jnp.deg2rad(pa_deg)
    dx, dy = x - x0, y - y0
    return dx * jnp.sin(pa) + dy * jnp.cos(pa), dx * jnp.cos(pa) - dy * jnp.sin(pa)


class Bending(NamedTuple):
    """GALFIT-style bending modes (see module docstring).

    m : tuple of ints (static); a : amplitudes in arcsec (ridge shift at
    |u| = R_e); pa_ref : degrees, places the seam (GALFIT: 0).
    """

    m: tuple
    a: list
    pa_ref: float


def bend_minor(u, v, R_e, pa_deg, bending):
    """Bent minor-axis coordinate v' (arcsec) for (u, v) from ``major_minor``."""
    s = jnp.sign(jnp.cos(jnp.deg2rad(pa_deg - bending.pa_ref)))
    shift = sum(a * (-s * u / R_e) ** m for m, a in zip(bending.m, bending.a))
    return v - s * shift


def elliptical_radius(x, y, x0, y0, q, pa_deg):
    """Elliptical radius R = sqrt(u^2 + (v/q)^2) in arcsec (see module docstring)."""
    pa = jnp.deg2rad(pa_deg)
    dx, dy = x - x0, y - y0
    u = dx * jnp.sin(pa) + dy * jnp.cos(pa)
    v = dx * jnp.cos(pa) - dy * jnp.sin(pa)
    return jnp.sqrt(u**2 + (v / q) ** 2)


def fourier_angle(x, y, x0, y0, q, pa_deg):
    """GALFIT's Fourier-mode angle theta_G = pi - atan2(v / q, u), in radians
    (see module docstring). Same inputs as ``elliptical_radius``."""
    pa = jnp.deg2rad(pa_deg)
    dx, dy = x - x0, y - y0
    u = dx * jnp.sin(pa) + dy * jnp.cos(pa)
    v = dx * jnp.cos(pa) - dy * jnp.sin(pa)
    return jnp.pi - jnp.arctan2(v / q, u)


# [M12e paused] class Spiral(NamedTuple):
    # [M12e paused] """GALFIT-style spiral rotation (see module docstring).

    # [M12e paused] kind : "power" or "log" (static); r_in, r_out : arcsec, 0 <= r_in < r_out;
    # [M12e paused] theta_out : degrees (either sign); shape : alpha ("power") or r_ws
    # [M12e paused] (arcsec, > 0; "log").
    # [M12e paused] """

    # [M12e paused] kind: str
    # [M12e paused] r_in: float
    # [M12e paused] r_out: float
    # [M12e paused] theta_out: float
    # [M12e paused] shape: float


# [M12e paused] def spiral_rotation(r, spiral):
    # [M12e paused] """theta(r) in degrees for circular radius r (arcsec); see module docstring."""
    # [M12e paused] th_rad = jnp.maximum(jnp.abs(jnp.deg2rad(spiral.theta_out)), 1e-6)  # B is singular at theta_out = 0
    # [M12e paused] B = 0.5 * jnp.log(0.46 / th_rad)
    # [M12e paused] x = (r - spiral.r_in) / (spiral.r_out - spiral.r_in) * (2.0 - B) + B
    # [M12e paused] t = 0.5 * (jnp.tanh(x) + 1.0)
    # [M12e paused] if spiral.kind == "power":
        # [M12e paused] h = (0.5 * (r / spiral.r_out + 1.0)) ** spiral.shape
    # [M12e paused] elif spiral.kind == "log":
        # [M12e paused] h = jnp.log(r / spiral.shape + 1.0) / jnp.log(spiral.r_out / spiral.shape + 1.0)
    # [M12e paused] else:
        # [M12e paused] raise ValueError(f"spiral kind must be 'power' or 'log', got {spiral.kind!r}")
    # [M12e paused] return spiral.theta_out * t * h


# [M12e paused] def spiral_pa(x, y, x0, y0, pa_deg, spiral):
    # [M12e paused] """PA_eff at positions (x, y): pa_deg + theta(|(x - x0, y - y0)|), or pa_deg if spiral is None."""
    # [M12e paused] if spiral is None:
        # [M12e paused] return pa_deg
    # [M12e paused] return pa_deg + spiral_rotation(jnp.hypot(x - x0, y - y0), spiral)


class CartesianModes(NamedTuple):
    """Fourier modes as c_m = a_m cos(m phi_m), s_m = a_m sin(m phi_m) (the
    fitted form, smooth at a_m = 0). Accepted wherever ``modes`` is."""

    m: tuple
    c: list
    s: list


def fourier_factor(theta, modes):
    """g(theta) = 1 + sum_k a_k cos(m_k (theta + phi_k)).

    Parameters
    ----------
    theta : array (radians)
        GALFIT angle from ``fourier_angle``.
    modes : tuple (m, a, phi_deg) or CartesianModes (m, c, s)
        m : tuple of ints (e.g. (1, 3, 4)); a, phi_deg : sequences of the same
        length (amplitudes, phases in degrees as GALFIT reports them).
        Cartesian: a cos(m (theta + phi)) = c cos(m theta) - s sin(m theta).

    Returns
    -------
    array with the shape of theta (dimensionless).
    """
    if isinstance(modes, CartesianModes):
        g = jnp.ones_like(theta)
        for k, mk in enumerate(modes.m):
            g = g + modes.c[k] * jnp.cos(mk * theta) - modes.s[k] * jnp.sin(mk * theta)
        return g
    m, a, phi_deg = modes
    g = jnp.ones_like(theta)
    for k, mk in enumerate(m):
        g = g + a[k] * jnp.cos(mk * (theta + jnp.deg2rad(phi_deg[k])))
    return g


def fourier_flux_factor(modes, n_theta=256):
    """Total flux of a Fourier-perturbed profile relative to the plain one:

        F / F_0 = (1 / 2 pi) * integral_0^{2 pi} g(theta)^(-2) d theta

    (derivation: substitute s = R g(theta) in the radial integral). Exact
    value for one mode m with amplitude a: (1 - a^2)^(-3/2).

    Parameters
    ----------
    modes : tuple (m, a, phi_deg) as in ``fourier_factor``, or None (-> 1).
    n_theta : int
        Number of angles used for the numerical integral.

    Returns
    -------
    float (JAX scalar, differentiable with respect to a and phi).
    """
    if modes is None:
        return 1.0
    # TODO(Erik): compute the integral above with jax.numpy (jnp).
    #   - Build n_theta angles over [0, 2 pi) and evaluate g with fourier_factor.
    #   - g is PERIODIC: think about which simple rule is best for a smooth
    #     periodic integrand, and whether the end point 2 pi should be included.
    #   - Return the AVERAGE of g^-2 over the circle (that is the 1 / 2 pi part).
    # Check your version with:  pytest tests/test_sersic.py -k flux_factor
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    g = fourier_factor(theta, modes)
    return jnp.mean(g**-2)


# [M12e paused] def sersic(x, y, x0, y0, I_e, R_e, n, q, pa_deg, modes=None, spiral=None):
def sersic(x, y, x0, y0, I_e, R_e, n, q, pa_deg, modes=None, bending=None):  # [M12e stand-in]
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
    modes : None or tuple (m, a, phi_deg)
        Optional GALFIT Fourier modes (see ``fourier_factor``). With modes,
        I_e is the brightness on the perturbed isophote R_F = R_e, and the
        total flux is no longer ``sersic_total_flux`` (M12b).
    bending : None or Bending
        Optional GALFIT bending modes (module docstring). The Fourier modes
        are measured in the bent frame. Bending does not change the flux.

    Returns
    -------
    array with the shape of x: surface brightness in the units of I_e.
    """
    # [M12e paused] pa_deg = spiral_pa(x, y, x0, y0, pa_deg, spiral)
    if bending is None:
        R = elliptical_radius(x, y, x0, y0, q, pa_deg)
        if modes is not None:
            R = R * fourier_factor(fourier_angle(x, y, x0, y0, q, pa_deg), modes)
    else:
        u, v = major_minor(x, y, x0, y0, pa_deg)
        v = bend_minor(u, v, R_e, pa_deg, bending)
        R = jnp.sqrt(u**2 + (v / q) ** 2)
        if modes is not None:
            R = R * fourier_factor(jnp.pi - jnp.arctan2(v / q, u), modes)
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


def sersic_radial_integral(X, I_e, R_e, n):
    """G(X) = integral_0^X I(R) R dR of a Sersic profile (closed form).

    With t = b_n (R/R_e)^(1/n):

        G(X) = I_e R_e^2 n e^(b_n) b_n^(-2n) Gamma(2n) * P(2n, b_n (X/R_e)^(1/n)),

    where P is the regularized lower incomplete gamma function. As X -> inf,
    2 pi q G(X) -> sersic_total_flux.

    Parameters
    ----------
    X : array
        Elliptical radius (arcsec), >= 0.
    I_e, R_e, n : float
        As in ``sersic``.

    Returns
    -------
    array like X, in (image units) x arcsec^2 per radian.
    """
    b = b_n(n)
    norm = I_e * R_e**2 * n * jnp.exp(b - 2.0 * n * jnp.log(b) + gammaln(2.0 * n))
    return norm * _P(2.0 * n, _gamma_argument(X, R_e, n, b))


def _P(a, x):
    """Regularized lower incomplete gamma P(a, x), with P(a, 0) = 0 set exactly.

    JAX's derivative of P (and Q) contains x^(a-1) = exp((a-1) ln x), which at
    x = 0 and a = 2n = 1 (n = 0.5, a common bound) is exp(0 * -inf) = NaN; in
    forward mode NaN * 0 still spreads to every parameter (found 2026-11-16
    when a 1A fit reached n = 0.5). x = 0 occurs for rays from the centre.
    """
    pos = x > 0
    return jnp.where(pos, gammainc(a, jnp.where(pos, x, 1.0)), 0.0)


def _Q(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P, with Q(a, 0) = 1 (see _P)."""
    pos = x > 0
    return jnp.where(pos, gammaincc(a, jnp.where(pos, x, 1.0)), 1.0)


def _gamma_argument(X, R_e, n, b):
    """x = b (X / R_e)^(1/n), with x = 0 at X = 0 and finite gradients there."""
    # X^(1/n) has an infinite derivative at X = 0 (for n > 1); evaluate the power
    # on a safe value there so gradients stay finite.
    positive = X > 0
    safe_X = jnp.where(positive, X, 1.0)
    return jnp.where(positive, b * (safe_X / R_e) ** (1.0 / n), 0.0)


def sersic_radial_integral_between(X_in, X_out, I_e, R_e, n):
    """G(X_out) - G(X_in), computed without loss of precision in the far tail.

    Near the centre the regularized lower incomplete gamma P is small and the
    difference P(x_out) - P(x_in) is accurate. Far out both P values are close
    to 1 and their difference loses all precision (e.g. 1 - 1e-12 minus
    1 - 1e-13), so there we use Q = 1 - P instead: Q(x_in) - Q(x_out).
    """
    b = b_n(n)
    a = 2.0 * n
    norm = I_e * R_e**2 * n * jnp.exp(b - 2.0 * n * jnp.log(b) + gammaln(a))
    x_in = _gamma_argument(X_in, R_e, n, b)
    x_out = _gamma_argument(X_out, R_e, n, b)
    lower = _P(a, x_out) - _P(a, x_in)
    upper = _Q(a, x_in) - _Q(a, x_out)
    # P < 1/2 for x below ~ the median (x < a); use Q-differences beyond it.
    return norm * jnp.where(x_in > a, upper, lower)


# [M12e paused] def sersic_polygon_flux(vertices, x0, y0, I_e, R_e, n, q, pa_deg, m=12, modes=None, spiral=None, spiral_split=4):
def sersic_polygon_flux(vertices, x0, y0, I_e, R_e, n, q, pa_deg, m=12, modes=None, bending=None,  # [M12e stand-in]
                        bend_split=4):  # [M12e stand-in]
    """Exact flux of a Sersic profile inside a convex polygon (e.g. a pixel).

    Method: shift to the profile centre, rotate to the major axis and stretch
    the minor axis by 1/q, so that the profile depends only on the distance R
    from the origin and area elements scale as dA = q dA'. Then

        F = q * integral [G(t_out) - G(t_in)] dphi,

    with G = ``sersic_radial_integral`` and t_in, t_out where the ray at angle
    phi enters and leaves the polygon (``forwardgal.geometry``). The radial
    integral is exact (closed form). The angle integral is split at the
    polygon's corner directions into smooth segments, each integrated with
    m-point Gauss-Legendre quadrature. For a pixel (4 segments) and m = 12,
    i.e. 48 angles, the relative error is <~1e-6 for all 0.5 <= n <= 8
    (tested against brute-force 2-D integration).

    Fourier modes: along a ray at stretched angle phi the profile is
    I(t g), g = g(theta_G = pi - phi). Substituting s = t g, the ray's
    contribution becomes [G(g t_out) - G(g t_in)] / g^2: still closed form.

    Parameters
    ----------
    vertices : (M, 2) array
        Polygon corners (theta_x, theta_y) in arcsec, +x east, +y north, in
        order (either winding direction). Must be convex and must not have a
        corner exactly at the profile centre.
    x0, y0, I_e, R_e, n, q, pa_deg : float
        Sersic parameters, as in ``sersic``.
    m : int
        Gauss-Legendre nodes per angular segment.
    modes : None or tuple (m, a, phi_deg)
        Optional Fourier modes (see ``fourier_factor``).
    bending : None or Bending
        APPROXIMATION: a 4-corner polygon is split into bend_split x
        bend_split sub-quadrilaterals; in each, the bend f(u) is replaced by
        its tangent line at the sub-quad's centre, a shear that keeps it a
        polygon of the same area (then exact). Error ~1/bend_split^2
        (measured in the tests).
    [M12e paused] spiral, spiral_split: k x k sub-quadrilaterals with the PA
        frozen at PA + theta(r_c) in each (error ~1/k^2, measured).

    Returns
    -------
    float
        Flux in (image units) x arcsec^2. Divide by the polygon area for the
        mean surface brightness (the model pixel value).
    """
    # [M12e paused] if spiral is not None:
        # [M12e paused] subs = _split_quad(vertices, spiral_split) if vertices.shape[0] == 4 else vertices[None]

        # [M12e paused] def one(v):
            # [M12e paused] pa_v = spiral_pa(jnp.mean(v[:, 0]), jnp.mean(v[:, 1]), x0, y0, pa_deg, spiral)
            # [M12e paused] return sersic_polygon_flux(v, x0, y0, I_e, R_e, n, q, pa_v, m=m, modes=modes)

        # [M12e paused] return jnp.sum(jax.vmap(one)(subs))
    if bending is not None:
        subs = _split_quad(vertices, bend_split) if vertices.shape[0] == 4 else vertices[None]
        s = jnp.sign(jnp.cos(jnp.deg2rad(pa_deg - bending.pa_ref)))

        def shift(u_):  # s * sum a_m (-s u / R_e)^m: the ridge, as in bend_minor
            return s * sum(a * (-s * u_ / R_e) ** mm for mm, a in zip(bending.m, bending.a))

        def one(vv):
            u, v = major_minor(vv[:, 0], vv[:, 1], x0, y0, pa_deg)
            uc = jnp.mean(u)
            v = v - (shift(uc) + jax.grad(shift)(uc) * (u - uc))  # tangent-line shear
            return _polygon_flux_uv(u, v, I_e, R_e, n, q, m, modes)

        return jnp.sum(jax.vmap(one)(subs))
    u, v = major_minor(vertices[:, 0], vertices[:, 1], x0, y0, pa_deg)
    return _polygon_flux_uv(u, v, I_e, R_e, n, q, m, modes)


def _polygon_flux_uv(u, v, I_e, R_e, n, q, m, modes):
    """Exact flux in the polygon with vertices (u, v) in the profile's own
    major/minor-axis frame (see ``sersic_polygon_flux``)."""
    stretched = jnp.stack([u, v / q], axis=1)

    # Gauss-Legendre nodes x_k and weights w_k on [-1, 1], mapped to each
    # segment [lo, hi]: phi = mid + half * x_k, weight = half * w_k.
    lo, hi = angular_segments(stretched)  # (M,)
    x_k, w_k = np.polynomial.legendre.leggauss(m)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    phi = mid[:, None] + half[:, None] * x_k[None, :]  # (M, m)
    weight = half[:, None] * w_k[None, :]

    t_in, t_out = ray_polygon_range(stretched, phi.ravel())
    if modes is None:
        dG = sersic_radial_integral_between(t_in, t_out, I_e, R_e, n)
    else:
        g = fourier_factor(jnp.pi - phi.ravel(), modes)  # theta_G = pi - phi
        dG = sersic_radial_integral_between(g * t_in, g * t_out, I_e, R_e, n) / g**2
    return q * jnp.sum(weight * dG.reshape(phi.shape))


def _split_quad(v, k):
    """(k*k, 4, 2) sub-quadrilaterals of a convex quadrilateral v (4, 2),
    corners in order, by bilinear interpolation of its corners (used for
    bending; also by the paused spiral code)."""
    t = jnp.linspace(0.0, 1.0, k + 1)
    a, b = t[:, None, None], t[None, :, None]  # (k+1, 1, 1), (1, k+1, 1)
    p = ((1 - a) * (1 - b) * v[0] + (1 - a) * b * v[1] + a * b * v[2] + a * (1 - b) * v[3])  # (k+1, k+1, 2)
    quads = jnp.stack([p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]], axis=2)  # (k, k, 4, 2)
    return quads.reshape(k * k, 4, 2)


def unit_flux_ie(R_e, n, q, pixscale, modes=None):
    """I_e that gives a total flux of 1 in pixel-sum units (sum of pixel values).

    A template rendered with this I_e has amplitude = total intrinsic flux in
    the units of the zeropoint, so m_tot = -2.5 log10(amplitude) + ZP.
    (I_e = A_pix / L(I_e = 1), A_pix = pixscale^2.) With Fourier modes, L is
    the flux of the PERTURBED profile (GALFIT's convention):
    L = sersic_total_flux * fourier_flux_factor(modes).
    """
    return pixscale**2 / (sersic_total_flux(1.0, R_e, n, q) * fourier_flux_factor(modes))


def mag_to_ie(m_tot, R_e, n, q, zeropoint, pixscale, modes=None):
    """Convert total magnitude to I_e.

    The sum of pixel values of a profile with total flux L is L / A_pix, with
    A_pix = pixscale^2 the pixel area in arcsec^2, and
    m_tot = -2.5 log10(L / A_pix) + zeropoint. Since L is proportional to I_e,

        I_e = 10^(-0.4 (m_tot - zeropoint)) * A_pix / L(I_e = 1).

    With Fourier modes, L includes fourier_flux_factor(modes) (as in GALFIT).
    """
    pixel_sum = 10.0 ** (-0.4 * (m_tot - zeropoint))
    return pixel_sum * pixscale**2 / (sersic_total_flux(1.0, R_e, n, q) * fourier_flux_factor(modes))


def ie_to_mag(I_e, R_e, n, q, zeropoint, pixscale, modes=None):
    """Convert I_e to total magnitude (inverse of mag_to_ie)."""
    pixel_sum = sersic_total_flux(I_e, R_e, n, q) * fourier_flux_factor(modes) / pixscale**2
    return -2.5 * jnp.log10(pixel_sum) + zeropoint
