"""Rendering surface-brightness models into pixel images (JAX).

Pixel values are pixel-AVERAGED surface brightness (image units, e.g. MJy/sr),
computed with a hybrid scheme:

1. Everywhere: uniform s x s subsampling, averaged per pixel.
2. In a K x K patch of pixels around each profile centre: the exact pixel
   integral (``sersic_polygon_flux``, Gauss-Legendre in angle, ~1e-6) divided
   by the pixel area, replacing the subsampled values there. Steep cores
   (large Sersic n) need this: point sampling is wrong by 1-20% next to the
   centre and unreliable in the centre pixel.

Measured accuracy of the uniform part (s = 5) outside a 7 x 7 patch:
<~1.5e-3 of the pixel value for the most compact profiles (R_e ~ 1 px),
smaller for larger R_e. ``render_sersic_exact`` integrates every pixel
exactly (slow, ~2 s for 167 x 167 px) and is used to check a fit's rendering
(``forwardgal.diagnostics.rendering_check``).

Grid conventions (unlensed, north-up): 0-based [row, col]; pixel (r, c)
covers [r - 0.5, r + 0.5] x [c - 0.5, c + 0.5]; theta from ``local_theta``
(+x east = towards smaller col, +y north = towards larger row).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from forwardgal.geometry import polygon_signed_area
from forwardgal.grids import bin_mean, local_theta, subpixel_centres
from forwardgal.profiles.sersic import sersic, sersic_polygon_flux


@dataclass(frozen=True)
class RenderSettings:
    """How pixel values are computed. Saved with every fit.

    Attributes
    ----------
    subsample : int
        Uniform subsampling factor s (s x s points per pixel).
    patch : int
        Unlensed only: side K of the exactly integrated square patch around
        the centre (odd).
    exact_pixels : int
        Lensed only: number N of pixels integrated exactly per component,
        chosen as the N pixels whose traced centres are nearest to the
        component centre IN THE SOURCE PLANE. In the image plane this is an
        ellipse stretched along the arc by the local magnification ratio, and
        it is shared automatically between multiple images of the centre.
        Measured (10x stretched image, n = 1-8): N = 200 gives <~6e-5 of the
        peak, similar to an unlensed 7 x 7 square patch.
    gl_nodes : int
        Gauss-Legendre nodes per angular segment in the exact pixel integral
        (4 segments per pixel, so 4 * gl_nodes angles).
    """

    subsample: int = 5
    patch: int = 7
    exact_pixels: int = 200
    gl_nodes: int = 12
    bend_split: int = 4  # exact pixels of bent components: k x k sub-quads (error ~1/k^2)
    # [M12e paused] spiral_split: int = 4  # exact pixels of spiral components: k x k sub-quads (error ~1/k^2)


def _pixel_corners(rows, cols, pixscale, row0, col0):
    """(..., 4, 2) theta corners of pixels at integer [rows, cols] (any shape)."""
    dr = jnp.array([-0.5, -0.5, 0.5, 0.5])
    dc = jnp.array([-0.5, 0.5, 0.5, -0.5])
    tx, ty = local_theta(rows[..., None] + dr, cols[..., None] + dc, pixscale, row0, col0)
    return jnp.stack([tx, ty], axis=-1)


# [M12e paused] def _exact_pixel_values(corners, pixscale, params, gl_nodes, modes=None, spiral=None, spiral_split=4):
EXACT_BATCH_VALUES = 2_000_000  # memory budget of the exact reference renderers (numbers per batch)


def _polygon_fluxes(quads, params, gl_nodes, modes=None, bending=None, bend_split=4):
    """Exact fluxes of many (N, 4, 2) polygons, in batches of bounded memory.

    One polygon needs ~4 gl_nodes numbers per intermediate array (x bend_split^2
    with bending); jax.lax.map processes as many polygons at a time as fit in
    EXACT_BATCH_VALUES. A plain vmap over all render pixels with bend_split 16
    needed 3 GB per intermediate array and crashed the machine (2026-11-16).
    """
    per = 4 * gl_nodes * (bend_split**2 if bending is not None else 1)
    batch = int(max(1, min(quads.shape[0], EXACT_BATCH_VALUES // per)))

    def one(v):
        return sersic_polygon_flux(v, *params, m=gl_nodes, modes=modes, bending=bending, bend_split=bend_split)

    return jax.lax.map(one, quads, batch_size=batch)


def _exact_pixel_values(corners, pixscale, params, gl_nodes, modes=None, bending=None, bend_split=4):  # [M12e stand-in]
    """Exact pixel-averaged values for (N, 4, 2) pixel corners -> (N,)."""
    # [M12e paused] (spiral=spiral, spiral_split=spiral_split were passed to sersic_polygon_flux here)
    flux = _polygon_fluxes(corners, params, gl_nodes, modes, bending, bend_split)
    return flux / pixscale**2


# [M12e paused] def _uniform_at_pixels(pix_rows, pix_cols, s, pixscale, row0, col0, params, modes=None, spiral=None):
def _uniform_at_pixels(pix_rows, pix_cols, s, pixscale, row0, col0, params, modes=None, bending=None):  # [M12e stand-in]
    """Uniformly subsampled pixel means at a list of pixels -> (P,) values.

    pix_rows, pix_cols : (P,) integer arrays (static) of 0-based pixel indices.
    """
    k = (jnp.arange(s) + 0.5) / s - 0.5  # subpixel offsets within a pixel
    dr, dc = jnp.meshgrid(k, k, indexing="ij")
    rows = pix_rows[:, None] + dr.ravel()[None, :]  # (P, s*s)
    cols = pix_cols[:, None] + dc.ravel()[None, :]
    tx, ty = local_theta(rows, cols, pixscale, row0, col0)
    # [M12e paused] return jnp.mean(sersic(tx, ty, *params, modes=modes, spiral=spiral), axis=1)
    return jnp.mean(sersic(tx, ty, *params, modes=modes, bending=bending), axis=1)  # [M12e stand-in]


def render_sersic(
    shape, pixscale, row0, col0, x0, y0, I_e, R_e, n, q, pa_deg, settings=RenderSettings(), pixels=None,
    # [M12e paused] modes=None, spiral=None,
# [M12e paused] ):
    modes=None, bending=None,  # [M12e stand-in]
):  # [M12e stand-in]
    """Unlensed Sersic image: hybrid of uniform subsampling and an exact patch.

    Parameters
    ----------
    shape : (ny, nx)
        Image shape (static).
    pixscale : float
        Pixel size in arcsec.
    row0, col0 : float
        0-based pixel position of theta = (0, 0).
    x0, y0, I_e, R_e, n, q, pa_deg : float
        Sersic parameters (see ``forwardgal.profiles.sersic.sersic``).
    settings : RenderSettings
    pixels : (rows, cols) integer arrays, optional
        If given, the uniform part is computed only at these pixels and all
        other pixels are 0 (e.g. the render pixels of a FitRegion, relative
        to its box). The exact patch is always computed in full.
    modes : None or tuple (m, a, phi_deg)
        Optional GALFIT Fourier modes (``forwardgal.profiles.sersic``).
    bending : None or Bending
        Optional GALFIT bending modes (exact pixels: settings.bend_split).

    Returns
    -------
    (ny, nx) array of pixel-averaged surface brightness, in units of I_e.
    """
    ny, nx = shape
    s, K = settings.subsample, settings.patch
    sersic_params = (x0, y0, I_e, R_e, n, q, pa_deg)

    # 1. Uniform subsampling, everywhere or only at the requested pixels.
    if pixels is None:
        rows, cols = subpixel_centres(ny, nx, s)
        tx, ty = local_theta(rows, cols, pixscale, row0, col0)
        # [M12e paused] image = bin_mean(sersic(tx, ty, *sersic_params, modes=modes, spiral=spiral), s)
        image = bin_mean(sersic(tx, ty, *sersic_params, modes=modes, bending=bending), s)  # [M12e stand-in]
    else:
        pr_, pc_ = (jnp.asarray(a) for a in pixels)
        # [M12e paused] values = _uniform_at_pixels(pr_, pc_, s, pixscale, row0, col0, sersic_params, modes, spiral)
        values = _uniform_at_pixels(pr_, pc_, s, pixscale, row0, col0, sersic_params, modes, bending)  # [M12e stand-in]
        image = jnp.zeros(shape).at[pr_, pc_].set(values)

    # 2. Pixel containing the centre (inverse of local_theta) and the K x K
    #    patch around it, shifted inward if it would leave the image. The
    #    patch size is fixed (JAX needs static shapes); its position is not.
    col_c = jnp.round(col0 - x0 / pixscale).astype(int)
    row_c = jnp.round(row0 + y0 / pixscale).astype(int)
    r_start = jnp.clip(row_c - K // 2, 0, ny - K)
    c_start = jnp.clip(col_c - K // 2, 0, nx - K)
    pr, pc = jnp.meshgrid(r_start + jnp.arange(K), c_start + jnp.arange(K), indexing="ij")

    # 3. Exact pixel integrals in the patch overwrite the subsampled values.
    corners = _pixel_corners(pr, pc, pixscale, row0, col0).reshape(K * K, 4, 2)
    # [M12e paused] patch = _exact_pixel_values(corners, pixscale, sersic_params, settings.gl_nodes, modes, spiral,
                                # [M12e paused] settings.spiral_split).reshape(K, K)
    patch = _exact_pixel_values(corners, pixscale, sersic_params, settings.gl_nodes, modes, bending,  # [M12e stand-in]
                                settings.bend_split).reshape(K, K)
    return jax.lax.dynamic_update_slice(image, patch, (r_start, c_start))


# [M12e paused] def render_sersic_exact(shape, pixscale, row0, col0, x0, y0, I_e, R_e, n, q, pa_deg, gl_nodes=12, modes=None,
                        # [M12e paused] spiral=None, spiral_split=4):
def render_sersic_exact(shape, pixscale, row0, col0, x0, y0, I_e, R_e, n, q, pa_deg, gl_nodes=12, modes=None,  # [M12e stand-in]
                        bending=None, bend_split=4):
    """Unlensed Sersic image with EVERY pixel integrated exactly (reference; slow).

    Same arguments and output as ``render_sersic``.
    """
    ny, nx = shape
    pr, pc = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    corners = _pixel_corners(pr, pc, pixscale, row0, col0).reshape(ny * nx, 4, 2)
    params = (x0, y0, I_e, R_e, n, q, pa_deg)
    # [M12e paused] return _exact_pixel_values(corners, pixscale, params, gl_nodes, modes, spiral, spiral_split).reshape(ny, nx)
    return _exact_pixel_values(corners, pixscale, params, gl_nodes, modes, bending, bend_split).reshape(ny, nx)  # [M12e stand-in]


# --- lensed rendering ----------------------------------------------------------


def _footprints(bcx, bcy):
    """(ny, nx, 4, 2) source-plane corners of every pixel from (ny+1, nx+1) corner grids."""
    p0 = jnp.stack([bcx[:-1, :-1], bcy[:-1, :-1]], axis=-1)  # corner [r, c]
    p1 = jnp.stack([bcx[:-1, 1:], bcy[:-1, 1:]], axis=-1)  # [r, c+1]
    p2 = jnp.stack([bcx[1:, 1:], bcy[1:, 1:]], axis=-1)  # [r+1, c+1]
    p3 = jnp.stack([bcx[1:, :-1], bcy[1:, :-1]], axis=-1)  # [r+1, c]
    return jnp.stack([p0, p1, p2, p3], axis=-2)


def _contains_point(quads, px, py):
    """(…) bool: does each (…, 4, 2) quadrilateral contain the point (px, py)?

    Same sign of all four edge cross products (either winding order).
    """
    a = quads
    e = jnp.roll(quads, -1, axis=-2) - a
    cross = e[..., 0] * (py - a[..., 1]) - e[..., 1] * (px - a[..., 0])
    return jnp.all(cross > 0, axis=-1) | jnp.all(cross < 0, axis=-1)


def nearest_source_pixels(quads, x0, y0, n_pixels):
    """Flat indices of the n_pixels pixels whose traced centres are nearest (x0, y0).

    quads : (ny, nx, 4, 2) traced footprints. The "traced centre" of a pixel is
    the mean of its four traced corners. Distances are measured in the source
    plane, so in the image plane the selected pixels form an ellipse stretched
    along the arc, around every image of (x0, y0).
    """
    centre = jnp.mean(quads, axis=-2)
    dist = jnp.hypot(centre[..., 0] - x0, centre[..., 1] - y0).ravel()
    _, flat = jax.lax.top_k(-dist, n_pixels)
    return flat


def render_lensed_sersic(grid, x0, y0, I_e, R_e, n, q, pa_deg, settings=RenderSettings(), pixels=None,
                         # [M12e paused] plane="source", modes=None, spiral=None):
                         plane="source", modes=None, bending=None):  # [M12e stand-in]
    """Sersic image on a traced grid: uniform part + exact pixels.

    plane = "source" (default): a lensed component; its parameters are in the
    source plane and it is evaluated at each pixel's traced beta.
    plane = "image": an unlensed (foreground / lens-plane) component; its
    parameters are in the image plane (theta) and it is evaluated at each
    pixel's theta, on the same pixels and subpixels. Everything else
    (exact pixels, PSF convolution, amplitudes) is identical.

    Pixel value = pixel average of I(beta(theta)) (surface brightness is
    conserved). Uniform part: mean of I at each pixel's s*s traced subpixels.
    Exact part: the settings.exact_pixels pixels nearest the centre in the
    source plane (``nearest_source_pixels``) get (flux in the traced
    footprint) / (footprint area), with the footprint = quadrilateral through
    the four traced pixel corners. This assumes the magnification is nearly
    constant across one pixel, which holds away from critical curves (masks
    avoid them).

    Parameters
    ----------
    grid : TracedGrid
        Pre-traced region (its subsample must equal settings.subsample).
        Close over it in jax.jit (it is constant data, not a parameter).
    x0, y0, I_e, R_e, n, q, pa_deg : float
        Source-plane Sersic parameters; (x0, y0) is a beta position (arcsec).
    settings : RenderSettings
    pixels : (rows, cols) integer arrays, optional
        If given, the uniform part is computed only there (0 elsewhere).
    modes : None or tuple (m, a, phi_deg)
        Optional GALFIT Fourier modes, in the component's own plane.
    bending : None or Bending
        Optional GALFIT bending modes, in the component's own plane.

    Returns
    -------
    (ny, nx) array of pixel-averaged surface brightness, in units of I_e.
    """
    if grid.subsample != settings.subsample:
        raise ValueError(f"grid traced with s={grid.subsample}, settings ask for s={settings.subsample}")
    ny, nx = grid.shape
    params = (x0, y0, I_e, R_e, n, q, pa_deg)
    sub_x, sub_y, corner_x, corner_y = grid.coordinates(plane)
    bsx, bsy = jnp.asarray(sub_x), jnp.asarray(sub_y)

    # 1. Uniform part: mean over the (traced or untraced) subpixels of each pixel.
    if pixels is None:
        # [M12e paused] image = jnp.mean(sersic(bsx, bsy, *params, modes=modes, spiral=spiral), axis=-1)
        image = jnp.mean(sersic(bsx, bsy, *params, modes=modes, bending=bending), axis=-1)  # [M12e stand-in]
    else:
        pr_, pc_ = (jnp.asarray(a) for a in pixels)
        # [M12e paused] values = jnp.mean(sersic(bsx[pr_, pc_], bsy[pr_, pc_], *params, modes=modes, spiral=spiral), axis=-1)
        values = jnp.mean(sersic(bsx[pr_, pc_], bsy[pr_, pc_], *params, modes=modes, bending=bending), axis=-1)  # [M12e stand-in]
        image = jnp.zeros((ny, nx)).at[pr_, pc_].set(values)

    # 2. Exact values for the pixels nearest the centre (in the component's plane).
    quads = _footprints(jnp.asarray(corner_x), jnp.asarray(corner_y))
    flat = nearest_source_pixels(quads, x0, y0, min(settings.exact_pixels, ny * nx))
    chosen = quads.reshape(ny * nx, 4, 2)[flat]
    # [M12e paused] flux = jax.vmap(lambda v: sersic_polygon_flux(v, *params, m=settings.gl_nodes, modes=modes, spiral=spiral,
                                                  # [M12e paused] spiral_split=settings.spiral_split))(chosen)
    flux = jax.vmap(lambda v: sersic_polygon_flux(v, *params, m=settings.gl_nodes, modes=modes,  # [M12e stand-in]
                                                  bending=bending, bend_split=settings.bend_split))(chosen)
    area = jnp.abs(jax.vmap(polygon_signed_area)(chosen))
    return image.ravel().at[flat].set(flux / area).reshape(ny, nx)


# [M12e paused] def render_lensed_sersic_exact(grid, x0, y0, I_e, R_e, n, q, pa_deg, gl_nodes=12, plane="source", modes=None,
                               # [M12e paused] spiral=None, spiral_split=4):
def render_lensed_sersic_exact(grid, x0, y0, I_e, R_e, n, q, pa_deg, gl_nodes=12, plane="source", modes=None,  # [M12e stand-in]
                               bending=None, bend_split=4):
    """Sersic with EVERY pixel integrated exactly over its footprint (slow; reference).

    plane as in ``render_lensed_sersic``.
    """
    ny, nx = grid.shape
    params = (x0, y0, I_e, R_e, n, q, pa_deg)
    _, _, corner_x, corner_y = grid.coordinates(plane)
    quads = _footprints(jnp.asarray(corner_x), jnp.asarray(corner_y)).reshape(ny * nx, 4, 2)
    # [M12e paused] (spiral=spiral, spiral_split=spiral_split were passed to sersic_polygon_flux here)
    flux = _polygon_fluxes(quads, params, gl_nodes, modes, bending, bend_split)
    area = jnp.abs(jax.vmap(polygon_signed_area)(quads))
    return (flux / area).reshape(ny, nx)


def source_plane_image(components, centre, size, pixscale, subsample=3):
    """Surface brightness of source-plane Sersic components on a regular beta grid.

    For PLOTTING the source model (not used in the fit). Each pixel is the
    mean of subsample x subsample point evaluations.

    Parameters
    ----------
    components : list of dict
        Sersic parameters (x0, y0, I_e, R_e, n, q, pa_deg) in the source plane.
    centre : (beta_x, beta_y)
        Centre of the grid (arcsec).
    size : float
        Side of the square grid (arcsec).
    pixscale : float
        Source-plane pixel size (arcsec).

    Returns
    -------
    image : (N, N) array, indexed [row, col] with +row = north and +col = WEST
        (east to the left, like the data), and extent (left, right, bottom,
        top) in beta for matplotlib's imshow(origin="lower").
    """
    n = int(round(size / pixscale))
    half = n * pixscale / 2
    k = (np.arange(n * subsample) + 0.5) / subsample  # fine pixel centres in pixel units
    bx = centre[0] + half - k * pixscale  # columns run west (beta_x decreasing)
    by = centre[1] - half + k * pixscale  # rows run north
    BX, BY = np.meshgrid(bx, by)
    total = sum(sersic(BX, BY, **c) for c in components)
    image = np.asarray(bin_mean(total, subsample))
    return image, (centre[0] + half, centre[0] - half, centre[1] - half, centre[1] + half)
