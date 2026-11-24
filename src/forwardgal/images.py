"""Multiple images of a source-plane point: solve beta = theta - alpha(theta).

Two steps:

1. Triangle search (numpy). Every cell of the deflection-map grid is split
   into two triangles. On grid nodes beta = theta - alpha is known exactly
   (no interpolation). The source-plane triangles that contain the target
   beta (all barycentric weights >= 0) mark the images; the same weights
   applied to the image-plane corners give a first guess for theta.
2. Newton refinement (JAX). theta <- theta - A^-1 [beta(theta) - beta_target]
   with the Jacobian A = d beta / d theta from automatic differentiation of
   the interpolated alpha (smooth for the default bicubic interpolation, so
   convergence is quadratic). Solutions from neighbouring triangles are
   merged.

Positions are theta (arcsec, +x east, +y north) in the deflection maps'
tangent plane; magnification mu = 1 / det A (negative = parity-flipped image).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LensedImage:
    """One image of a source-plane point.

    Attributes
    ----------
    theta : (float, float)
        Image position (arcsec).
    mu : float
        Signed magnification 1 / det A (negative: mirror-imaged).
    residual : float
        |beta(theta) - beta_target| after refinement (arcsec).
    """

    theta: tuple
    mu: float
    residual: float


def _node_beta(field):
    """theta and beta (= theta - alpha, exact on nodes) on the full map grid."""
    ny, nx = field.alpha_x.shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    tx, ty = field.pixel_to_theta(rows, cols)
    tx, ty = np.asarray(tx, float), np.asarray(ty, float)
    return tx, ty, tx - field.alpha_x, ty - field.alpha_y


def triangle_candidates(field, beta_x, beta_y):
    """First guesses for all images of (beta_x, beta_y): (N, 2) array of theta.

    Each grid cell gives two triangles (corners 00-10-11 and 00-11-01, where
    the digits are row and column offsets). A triangle is a candidate if the
    target lies inside its source-plane image.
    """
    tx, ty, bx, by = _node_beta(field)
    s = slice(None, -1), slice(None, -1)
    c00 = (tx[s], ty[s], bx[s], by[s])
    c10 = (tx[1:, :-1], ty[1:, :-1], bx[1:, :-1], by[1:, :-1])
    c11 = (tx[1:, 1:], ty[1:, 1:], bx[1:, 1:], by[1:, 1:])
    c01 = (tx[:-1, 1:], ty[:-1, 1:], bx[:-1, 1:], by[:-1, 1:])
    guesses = []
    for a, b, c in [(c00, c10, c11), (c00, c11, c01)]:
        # Barycentric weights (wa, wb, wc) of the target in source triangle (a, b, c).
        x1, y1 = b[2] - a[2], b[3] - a[3]
        x2, y2 = c[2] - a[2], c[3] - a[3]
        xp, yp = beta_x - a[2], beta_y - a[3]
        det = x1 * y2 - x2 * y1
        with np.errstate(divide="ignore", invalid="ignore"):
            wb = (xp * y2 - x2 * yp) / det
            wc = (x1 * yp - xp * y1) / det
        wa = 1.0 - wb - wc
        eps = 1e-9
        inside = (wa >= -eps) & (wb >= -eps) & (wc >= -eps) & np.isfinite(wb) & np.isfinite(wc)
        gx = wa * a[0] + wb * b[0] + wc * c[0]
        gy = wa * a[1] + wb * b[1] + wc * c[1]
        guesses.append(np.column_stack([gx[inside], gy[inside]]))
    return np.vstack(guesses)


def magnification(field, tx, ty):
    """Signed point magnification mu = 1 / det A at image position theta = (tx, ty) (arcsec).

    A = d beta / d theta (the lens equation's Jacobian, by JAX autodiff of
    field.backtrace); negative mu: a parity-flipped (mirror) image.
    """
    def lens_eq(t):
        bx, by = field.backtrace(t[0], t[1])
        return jnp.stack([bx, by])

    return float(1.0 / jnp.linalg.det(jax.jacfwd(lens_eq)(jnp.array([float(tx), float(ty)]))))


def _newton(field, beta_target, theta0, steps):
    """Refine one image position with ``steps`` Newton iterations (JAX)."""

    def lens_eq(t):
        bx, by = field.backtrace(t[0], t[1])
        return jnp.stack([bx, by])

    jac = jax.jacfwd(lens_eq)

    def step(t, _):
        A = jac(t)
        return t - jnp.linalg.solve(A, lens_eq(t) - beta_target), None

    t, _ = jax.lax.scan(step, theta0, None, length=steps)
    return t, jnp.linalg.det(jac(t)), jnp.linalg.norm(lens_eq(t) - beta_target)


def find_images(field, beta_x, beta_y, newton_steps=15, merge_tol=1e-4, max_residual=1e-8):
    """All images of the source-plane point (beta_x, beta_y) within the maps.

    Parameters
    ----------
    field : DeflectionField
    beta_x, beta_y : float
        Source position (arcsec).
    newton_steps : int
        Newton iterations per candidate (quadratic convergence: ~5 suffice
        from the triangle guesses; extra steps cost little).
    merge_tol : float
        Solutions closer than this (arcsec) are the same image.
    max_residual : float
        Solutions with |beta(theta) - beta_target| above this (arcsec) are
        dropped (e.g. Newton left the map or did not converge).

    Returns
    -------
    list of LensedImage, sorted by |mu| (brightest first).
    """
    guesses = triangle_candidates(field, beta_x, beta_y)
    if len(guesses) == 0:
        return []
    target = jnp.array([beta_x, beta_y], dtype=float)
    refine = jax.jit(jax.vmap(lambda t0: _newton(field, target, t0, newton_steps)))
    thetas, dets, residuals = (np.asarray(a) for a in refine(jnp.asarray(guesses)))

    images = []
    for t, d, r in zip(thetas, dets, residuals):
        if not (np.isfinite(r) and r < max_residual):
            continue
        if any(np.hypot(*(t - np.array(im.theta))) < merge_tol for im in images):
            continue
        images.append(LensedImage(theta=(float(t[0]), float(t[1])), mu=float(1.0 / d), residual=float(r)))
    return sorted(images, key=lambda im: -abs(im.mu))
