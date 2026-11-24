"""Plotting helpers (matplotlib). Not used in fits.

Coordinates are theta / beta in arcsec with +x = east; callers should put
east on the left (``ax.set_xlim(right, left)``) to match the data.
"""

from __future__ import annotations

import numpy as np


def corner_mask(mask):
    """(ny+1, nx+1) bool: True for pixel corners that touch at least one mask pixel."""
    ny, nx = mask.shape
    out = np.zeros((ny + 1, nx + 1), bool)
    out[:-1, :-1] |= mask
    out[:-1, 1:] |= mask
    out[1:, :-1] |= mask
    out[1:, 1:] |= mask
    return out


def grid_lines(corner_x, corner_y, keep=None, every=1):
    """Lines of a (traced or untraced) pixel grid, as (x, y) arrays with NaN gaps.

    Every ``every``-th row of corners and every ``every``-th column of corners
    becomes one polyline; corners where ``keep`` is False are replaced by NaN,
    so matplotlib breaks the line there (e.g. outside the mask).

    Returns
    -------
    rows, cols : lists of (x, y) array pairs
    """
    cx, cy = np.asarray(corner_x, float), np.asarray(corner_y, float)
    if keep is not None:
        cx, cy = np.where(keep, cx, np.nan), np.where(keep, cy, np.nan)
    rows = [(cx[i], cy[i]) for i in range(0, cx.shape[0], every)]
    cols = [(cx[:, j], cy[:, j]) for j in range(0, cx.shape[1], every)]
    return rows, cols


def plot_traced_grid(ax, grid, mask=None, every=6, plane="source", row_kw=None, col_kw=None):
    """Draw the pixel grid of a TracedGrid, optionally only inside a mask.

    plane = "source": the grid traced back to the source plane (beta), which
    shows the lens distortion; plane = "image": the same grid lines in the
    image plane (theta), for side-by-side comparison.

    Rows of pixel corners (constant image row, i.e. running east-west in the
    image) and columns (running north-south) get different colours by default,
    so the two families can be followed from one plane to the other.
    """
    if plane == "source":
        cx, cy = grid.beta_corner_x, grid.beta_corner_y
    elif plane == "image":
        cx, cy = grid.theta_corner_x, grid.theta_corner_y
    else:
        raise ValueError(f"plane must be 'source' or 'image', got {plane!r}")
    keep = corner_mask(np.asarray(mask, bool)) if mask is not None else None
    rows, cols = grid_lines(cx, cy, keep, every)
    row_kw = dict(color="C0", lw=0.6) | (row_kw or {})
    col_kw = dict(color="C1", lw=0.6) | (col_kw or {})
    for k, (x, y) in enumerate(rows):
        ax.plot(x, y, **(row_kw | ({"label": "image rows (E-W lines)"} if k == 0 else {})))
    for k, (x, y) in enumerate(cols):
        ax.plot(x, y, **(col_kw | ({"label": "image columns (N-S lines)"} if k == 0 else {})))


def plot_fit_summary(model, result, field=None, quasar_beta=None, zeropoint=None, src_size=2.0,
                     src_pixscale=0.005, title="", path=None, errors=None, extra_text=""):
    """One-page figure of a fit: data + mask, model, residuals, source plane, numbers.

    Image-plane panels are in theta (arcsec, east on the left). The source-plane
    panel shows the fitted SOURCE-plane Sersic components on a regular beta grid
    (east on the left), the caustics (if ``field`` is given), the quasar (if
    ``quasar_beta``) and the mask traced to the source plane.

    Parameters
    ----------
    model : ForwardModel
    result : FitResult
    field : DeflectionField, optional
    quasar_beta : (float, float), optional
    zeropoint : float, optional
        For AB magnitudes of the fitted fluxes.
    path : str or Path, optional
        Save the PNG here.
    errors : LaplaceErrors, optional
        If given, its table (best +- sigma) replaces the plain parameter list.
    extra_text : str
        Appended to the text panel (e.g. the rendering-check results).

    Returns
    -------
    matplotlib Figure
    """
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
    from scipy.ndimage import binary_closing

    # [M12e paused] from forwardgal.components import modes_cs, spiral_of
    from forwardgal.components import bending_of, modes_cs  # [M12e stand-in]
    from forwardgal.profiles.sersic import unit_flux_ie
    from forwardgal.render import source_plane_image

    u = jnp.asarray(result.u)
    data, sigma, mask = np.asarray(model.data), np.asarray(model.sigma), np.asarray(model.mask, bool)
    mdl = np.asarray(model.model_image(u))
    resid = np.where(mask, (data - mdl) / sigma, np.nan)
    c2, dof = model.chi2_dof(u)
    tx, ty = model.grid.theta_corner_x, model.grid.theta_corner_y
    ext = (tx[0, 0], tx[0, -1], ty[0, 0], ty[-1, 0])
    rendered = np.zeros(mask.shape, bool)
    rendered[model.pixels] = True

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    norm = ImageNormalize(data[mask], interval=ZScaleInterval(), stretch=AsinhStretch())
    panels = [
        (gs[0, 0], data, "data, mask (white) and render region (orange)", dict(cmap="gray", norm=norm)),
        (gs[0, 1], mdl, "best-fit model (all components, PSF-convolved)", dict(cmap="gray", norm=norm)),
        (gs[0, 2], resid, f"(data - model) / sigma in the mask;  chi2/dof = {c2:.0f}/{dof} = {c2 / dof:.3f}",
         dict(cmap="RdBu_r", vmin=-5, vmax=5)),
    ]
    for spec, arr, ttl, kw in panels:
        ax = fig.add_subplot(spec)
        im = ax.imshow(arr, origin="lower", extent=ext, **kw)
        ax.contour(mask, levels=[0.5], colors="w", linewidths=1.0, linestyles="--", extent=ext, origin="lower")
        if arr is data:
            ax.contour(rendered, levels=[0.5], colors="orange", linewidths=0.8, extent=ext, origin="lower")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("theta_x [arcsec, +east]")
        ax.set_ylabel("theta_y [arcsec, +north]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Source plane.
    ax = fig.add_subplot(gs[1, 0])
    src = [c for c in model.params.components if c.kind == "sersic" and c.plane == "source"]
    centre = quasar_beta
    if centre is None and src:
        centre = (float(result.values[src[0].name]["x0"]), float(result.values[src[0].name]["y0"]))
    if centre is not None:
        comps = []
        for c in src:
            v = {k: float(x) for k, x in result.values[c.name].items()}
            modes = modes_cs(v, c.modes)  # None without Fourier modes
            ie = result.amplitudes[c.name] * float(unit_flux_ie(v["R_e"], v["n"], v["q"], model.pixscale,
                                                                modes=modes))
            # the profile takes q and PA (results carry both e1, e2 and the derived q, PA)
            keep = ("x0", "y0", "R_e", "n", "q", "pa_deg")
            # [M12e paused] comps.append(dict({k: v[k] for k in keep}, I_e=ie, modes=modes, spiral=spiral_of(v, c.spiral_kind)))
            comps.append(dict({k: v[k] for k in keep}, I_e=ie, modes=modes, bending=bending_of(v, c)))  # [M12e stand-in]
        if comps:
            img, sext = source_plane_image(comps, centre=centre, size=src_size, pixscale=src_pixscale)
            im = ax.imshow(np.log10(np.maximum(img, 1e-3 * img.max())), origin="lower", extent=sext, cmap="magma")
            fig.colorbar(im, ax=ax, fraction=0.046, label="log10 I [image units]")
        else:
            sext = (centre[0] + src_size / 2, centre[0] - src_size / 2, centre[1] - src_size / 2, centre[1] + src_size / 2)
        bx, by = model.grid.beta_sub_x.mean(axis=-1)[mask], model.grid.beta_sub_y.mean(axis=-1)[mask]
        nb = int(round(src_size / src_pixscale))
        ex = np.linspace(sext[1], sext[0], nb + 1)
        ey = np.linspace(sext[2], sext[3], nb + 1)
        cov = binary_closing(np.histogram2d(by, bx, bins=[ey, ex])[0] > 0, iterations=3)
        ax.contour(cov[:, ::-1], levels=[0.5], colors="c", linewidths=1.2, extent=sext, origin="lower")
        ax.plot([], [], "c-", label="mask, traced")
        if field is not None:
            from forwardgal.critical import caustics, critical_curves

            for k, c in enumerate(caustics(field, critical_curves(field))):
                ax.plot(c[:, 0], c[:, 1], "w-", lw=0.8, label="caustics" if k == 0 else None)
        if quasar_beta is not None:
            ax.plot(*quasar_beta, "*", color="lime", ms=12, label="quasar")
        ax.set_xlim(sext[0], sext[1])
        ax.set_ylim(sext[2], sext[3])
        ax.legend(loc="lower left", fontsize=7)
    ax.set_title("source plane: fitted source-plane components", fontsize=9)
    ax.set_xlabel("beta_x [arcsec, +east]")
    ax.set_ylabel("beta_y [arcsec, +north]")
    ax.set_aspect("equal")

    # Numbers.
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    lines = [title, f"chi2/dof = {c2:.1f} / {dof} = {c2 / dof:.3f}", ""]
    if errors is not None:
        lines += [errors.describe(zeropoint=zeropoint), "", extra_text]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=110, bbox_inches="tight")
        return fig
    for c in model.params.components:
        v = result.values[c.name]
        pars = ", ".join(f"{k}={float(x):.4g}" for k, x in v.items())
        lines.append(f"{c.name} ({c.kind}, {c.plane} plane): {pars}")
    lines.append("")
    for n, a in result.amplitudes.items():
        mag = f"  ->  {-2.5 * np.log10(a) + zeropoint:.2f} AB" if (zeropoint is not None and a > 0
                                                                   and model._kind(n) != "sky") else ""
        unit = "MJy/sr" if model._kind(n) == "sky" else "pixel-sum flux"
        lines.append(f"amplitude {n}: {a:.5g} [{unit}]{mag}")
    lines += ["", extra_text]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig
