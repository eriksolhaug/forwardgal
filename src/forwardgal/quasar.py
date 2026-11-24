"""The quasar: a point source (delta function) in the source plane.

Modes (see MILESTONES, M6b/M7):
- image-fixed  : the user gives the image position (RA, Dec). theta is fixed,
                 the source position beta = theta - alpha(theta) is derived once
                 and reported; only the flux is free. With a PSF made for this
                 exact position, the PSF is placed without any fractional shift.
- source-fixed : beta given; image positions from the lens equation (M7).
- free         : beta free, optional small per-image offsets (M7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageFixedQuasar:
    """A quasar image whose sky position is fixed.

    Attributes
    ----------
    ra, dec : float
        Fixed image position (deg).
    theta : (float, float)
        Image-plane position (arcsec, +x east, +y north).
    beta : (float, float)
        Derived source-plane position beta = theta - alpha(theta) (arcsec).
    """

    ra: float
    dec: float
    theta: tuple
    beta: tuple

    def grid_position(self, image_wcs, origin=(0, 0)):
        """Fractional 0-based (row, col) on a grid whose [0, 0] is image pixel ``origin``."""
        col, row = image_wcs.wcs_world2pix(self.ra, self.dec, 0)
        return float(row) - origin[0], float(col) - origin[1]

    def describe(self):
        return "\n".join([
            f"quasar (image-fixed): RA, Dec = {self.ra:.7f}, {self.dec:.7f}",
            f"  theta = ({self.theta[0]:+.4f}, {self.theta[1]:+.4f}) arcsec",
            f"  beta  = ({self.beta[0]:+.4f}, {self.beta[1]:+.4f}) arcsec  (derived, fixed)",
        ])


def image_fixed_quasar(ra, dec, field):
    """Fix a quasar image at (ra, dec) and derive its source position through ``field``."""
    tx, ty = field.sky_to_theta(ra, dec)
    bx, by = field.backtrace(tx, ty)
    if not np.isfinite(float(bx)):
        raise ValueError("quasar position lies outside the deflection maps")
    return ImageFixedQuasar(
        ra=float(ra), dec=float(dec), theta=(float(tx), float(ty)), beta=(float(bx), float(by))
    )


def snap_to_psf(grid_rc, psf_ref, psf_centre, max_snap=0.1):
    """Snap a grid position so a PSF made for it needs no fractional shift.

    A PSF made for one specific image (e.g. 1A) already contains that image's
    sub-pixel position: it is exact when its pixels line up with the image's.
    A measured position differs from that by measurement noise (~0.03 px for
    1A); shifting the PSF by such a tiny amount would only add sinc ringing
    (see psf._odd_fast_len). This returns the nearest position at which the
    stamp aligns with whole pixels, and the snap distance in pixels.

    Raises ValueError if the snap exceeds ``max_snap`` pixels: then the PSF
    was not made for this position. max_snap=None: no limit ("trust the PSF":
    the PSF is taken to be made for this quasar; user's choice for the other
    bands, 2026-11-16); the returned distance shows how far it moved.
    """
    off = np.asarray(psf_ref, float) - np.asarray(psf_centre, float)
    frac = np.asarray(grid_rc, float) - off
    snapped = np.round(frac) + off
    dist = float(np.hypot(*(snapped - np.asarray(grid_rc, float))))
    if max_snap is not None and dist > max_snap:
        raise ValueError(f"snapping would move the image by {dist:.3f} px (> {max_snap} px): "
                         "the PSF does not seem to be made for this position")
    return (float(snapped[0]), float(snapped[1])), dist


# --- shared: theta -> model-grid pixel, and unit point-source images ------------

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from forwardgal.images import _newton, find_images  # noqa: E402
from forwardgal.psf import make_point_source_renderer  # noqa: E402


@dataclass(frozen=True)
class GridMapping:
    """Affine map theta (arcsec) -> fractional 0-based (row, col) on a model grid.

    The exact chain is theta -> RA, Dec (deflection-map TAN plane) -> pixel
    (image WCS). Over a few arcsec both projections are linear to ~1e-9 px,
    so it is replaced by [row, col] = rc0 + M (theta - theta0), which works
    inside JAX. Built by ``grid_mapping`` from finite differences; its
    accuracy is checked against astropy in the tests.
    """

    theta0: tuple
    rc0: tuple
    M: np.ndarray  # 2 x 2: d(row, col) / d(theta_x, theta_y)

    def __call__(self, tx, ty):
        dx, dy = tx - self.theta0[0], ty - self.theta0[1]
        row = self.rc0[0] + self.M[0, 0] * dx + self.M[0, 1] * dy
        col = self.rc0[1] + self.M[1, 0] * dx + self.M[1, 1] * dy
        return row, col


def theta_to_grid(image_wcs, field, origin):
    """The exact map theta (arcsec) -> fractional 0-based [row, col] on a grid.

    theta -> RA, Dec through the deflection maps' TAN plane (field), then RA,
    Dec -> image pixel through the image WCS; ``origin`` = the image pixel
    [row, col] of the grid's [0, 0]. Not for JAX (astropy): see GridMapping.
    """
    def to_grid(tx, ty):
        ra, dec = field.theta_to_sky(tx, ty)
        col, row = image_wcs.wcs_world2pix(ra, dec, 0)  # 0-based (x = col, y = row)
        return np.array([float(row) - origin[0], float(col) - origin[1]])

    return to_grid


def grid_mapping(image_wcs, field, origin, shape):
    """GridMapping for a grid of ``shape`` whose [0, 0] is image pixel ``origin``."""
    to_grid = theta_to_grid(image_wcs, field, origin)

    # theta at the grid centre, then the 2 x 2 derivative by central differences.
    rc_c = ((shape[0] - 1) / 2 + origin[0], (shape[1] - 1) / 2 + origin[1])
    ra, dec = image_wcs.wcs_pix2world(rc_c[1], rc_c[0], 0)
    t0 = tuple(float(v) for v in field.sky_to_theta(ra, dec))
    h = 0.01  # arcsec
    M = np.column_stack([
        (to_grid(t0[0] + h, t0[1]) - to_grid(t0[0] - h, t0[1])) / (2 * h),
        (to_grid(t0[0], t0[1] + h) - to_grid(t0[0], t0[1] - h)) / (2 * h),
    ])
    return GridMapping(theta0=t0, rc0=tuple(to_grid(*t0)), M=M)


# --- a locked point component's template, with its own PSF stamp (2026-11-22) ----


def psf_reference(kernel, reference="centroid"):
    """Where in a PSF stamp the point it was made for lies: 0-based [row, col].

    kernel : (ky, kx) PSF, normalized to unit sum (psf.load_psf does this)
    reference : "centroid" (light centroid: right for a PSF made for one
        specific quasar image, e.g. 1A), "centre" (GALFIT centre
        [ky // 2, kx // 2]), or an explicit (row, col) in stamp pixels.
    """
    kernel = np.asarray(kernel)
    if isinstance(reference, str):
        if reference == "centroid":
            yy, xx = np.indices(kernel.shape)
            return (float((kernel * yy).sum()), float((kernel * xx).sum()))  # unit sum: no division
        if reference == "centre":
            return (float(kernel.shape[0] // 2), float(kernel.shape[1] // 2))
        raise ValueError(f"PSF reference must be 'centroid', 'centre' or (row, col), got {reference!r}")
    row, col = reference
    return (float(row), float(col))


@dataclass(frozen=True)
class PlacedPoint:
    """A point source placed on a grid by ``place_point``."""

    template: np.ndarray  # (ny, nx) unit-flux image of the point (PSF-convolved: it IS the PSF)
    rc: tuple  # snapped 0-based [row, col] on the grid
    snap: float  # how far the position was moved to get there (px)


def place_point(rc, kernel, shape, reference="centroid", max_snap=0.1):
    """Unit-flux image of a point source at grid position ``rc``, drawn with ``kernel``.

    rc : fractional 0-based [row, col] on the grid (e.g. from theta_to_grid)
    kernel : the PSF stamp for THIS point (each point may have its own)
    shape : (ny, nx) grid shape (the render box)
    reference, max_snap : see psf_reference and snap_to_psf. The position
        is snapped so the stamp lands on whole pixels (no sub-pixel shift);
        a snap larger than max_snap px raises ValueError (None: no limit).
    """
    ref = psf_reference(kernel, reference)
    centre = (np.shape(kernel)[0] // 2, np.shape(kernel)[1] // 2)
    rc_snapped, dist = snap_to_psf(rc, ref, centre, max_snap=max_snap)
    template = np.asarray(make_point_source_renderer(kernel, shape, ref)(*rc_snapped))
    return PlacedPoint(template=template, rc=rc_snapped, snap=dist)


def _in_region(rc, shape, pad):
    return -pad <= rc[0] <= shape[0] - 1 + pad and -pad <= rc[1] <= shape[1] - 1 + pad


# --- source-fixed mode --------------------------------------------------------------


@dataclass(frozen=True)
class SourceFixedQuasar:
    """Quasar at a fixed source position; image positions from the lens equation.

    Attributes
    ----------
    beta : (float, float)
        Fixed source position (arcsec).
    images : list of LensedImage
        All images of beta in the deflection maps (``find_images``).
    in_region : list of int
        Indices into ``images`` of the images that fall on the model grid
        (padded by the PSF half-width); only these are rendered.
    grid_positions : list of (row, col)
        Their fractional grid positions.
    """

    beta: tuple
    images: list
    in_region: list
    grid_positions: list

    def unit_images(self, kernels, shape, psf_refs=None):
        """(n_in_region, ny, nx) unit-flux point-source images (precomputed once).

        kernels : one normalized PSF per image in ``in_region`` (or a single PSF
        for all). psf_refs : matching stamp reference points (default: centres).
        """
        n = len(self.in_region)
        kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels] * n
        psf_refs = psf_refs if psf_refs is not None else [None] * n
        out = [
            make_point_source_renderer(k, shape, ref)(*rc)
            for k, ref, rc in zip(kernels, psf_refs, self.grid_positions)
        ]
        return jnp.stack(out) if out else jnp.zeros((0, *shape))

    def describe(self):
        lines = [f"quasar (source-fixed): beta = ({self.beta[0]:+.4f}, {self.beta[1]:+.4f}) arcsec, "
                 f"{len(self.images)} images in the lens model"]
        for k, im in enumerate(self.images):
            where = "on the model grid" if k in self.in_region else "outside the model grid"
            lines.append(f"  image {k}: theta = ({im.theta[0]:+.3f}, {im.theta[1]:+.3f}), mu = {im.mu:+.2f}  ({where})")
        return "\n".join(lines)


def source_fixed_quasar(beta_x, beta_y, field, mapping, shape, pad=70):
    """Find all images of (beta_x, beta_y) and keep those on the model grid.

    pad : images up to this many pixels outside the grid are kept, because
    their PSF wings still reach it (default: the F115W PSF half-width).
    """
    images = find_images(field, beta_x, beta_y)
    keep, pos = [], []
    for k, im in enumerate(images):
        rc = tuple(float(v) for v in mapping(*im.theta))
        if _in_region(rc, shape, pad):
            keep.append(k)
            pos.append(rc)
    return SourceFixedQuasar(beta=(float(beta_x), float(beta_y)), images=images, in_region=keep, grid_positions=pos)


# --- free mode ------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeQuasar:
    """Quasar whose source position (and optional per-image offsets) are fitted.

    Image positions follow the source: theta_i(beta) solves the lens equation by
    ``newton_steps`` Newton iterations inside JAX, starting from the images found
    at the starting source position ``beta0``. Differentiating through the
    converged iterations gives d theta_i / d beta = A(theta_i)^-1. Valid while
    beta does not cross a caustic (the number of images must not change);
    ``image_positions`` also returns the Newton residuals to check this.

    Attributes
    ----------
    beta0 : (float, float)
        Starting source position (arcsec).
    theta0 : (n, 2) array
        Image positions at beta0 for the images on the model grid.
    """

    beta0: tuple
    theta0: np.ndarray
    field: object
    mapping: GridMapping
    newton_steps: int = 8

    @property
    def n_images(self):
        return len(self.theta0)

    def image_positions(self, beta, offsets=None):
        """theta_i(beta) + offsets_i, and Newton residuals |beta(theta_i) - beta| (arcsec).

        beta : (2,) array. offsets : (n, 2) array of per-image theta offsets
        (arcsec), or None. Returns (n, 2) positions and (n,) residuals.
        """
        beta = jnp.asarray(beta, dtype=float)
        solve = jax.vmap(lambda t0: _newton(self.field, beta, t0, self.newton_steps))
        theta, _, residual = solve(jnp.asarray(self.theta0))
        if offsets is not None:
            theta = theta + jnp.asarray(offsets)
        return theta, residual

    def unit_images(self, beta, offsets, renderers):
        """(n, ny, nx) unit-flux point-source images; ``renderers`` from
        ``make_point_source_renderer``, one per image (or a single one)."""
        theta, _ = self.image_positions(beta, offsets)
        renderers = renderers if isinstance(renderers, (list, tuple)) else [renderers] * self.n_images
        out = []
        for k, render in enumerate(renderers):
            row, col = self.mapping(theta[k, 0], theta[k, 1])
            out.append(render(row, col))
        return jnp.stack(out)


def free_quasar(beta0_x, beta0_y, field, mapping, shape, pad=70, newton_steps=8):
    """Set up the free mode from a starting source position."""
    sf = source_fixed_quasar(beta0_x, beta0_y, field, mapping, shape, pad)
    theta0 = np.array([sf.images[k].theta for k in sf.in_region]).reshape(-1, 2)
    return FreeQuasar(beta0=(float(beta0_x), float(beta0_y)), theta0=theta0, field=field,
                      mapping=mapping, newton_steps=newton_steps)
