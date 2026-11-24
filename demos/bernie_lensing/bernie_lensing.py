"""Lens a colour picture through the cluster deflection field, as an animated GIF.

A picture (RGBA PNG; transparent pixels are empty sky) is placed in the
source plane and slides from outside the tangential caustic to the source
position of the quasar QSO 1 (beta of image 1A). Left panel: the source plane
with the caustics (cyan). Right panel: the image plane with the critical
curves (red), where the picture is split into several magnified, distorted
images. --clean: both panels, pictures only; --image-only: the image plane,
picture only.

Lensing conserves surface brightness, so each image-plane pixel shows the
picture's colour at its traced source position:

    I_image(theta) = I_source(beta(theta)),   beta = theta - alpha(theta).

beta(theta) is computed once on a supersampled grid (forwardgal's
DeflectionField.backtrace); every frame then only moves the picture.

Coordinates: theta, beta in arcsec relative to the lens-model reference
position; +x east, +y north; both panels are shown north up, east LEFT.
The deflection maps are for z = 1.524 = the quasar redshift (no rescaling).

Run from the repository root:
    python demos/bernie_lensing/bernie_lensing.py [--height 2.0] [--frames 90] [--clean | --image-only] [--out file.gif]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

import forwardgal  # noqa: E402, F401
from forwardgal.critical import caustics, critical_curves  # noqa: E402
from forwardgal.lens import load_deflection  # noqa: E402
from forwardgal.quasar import image_fixed_quasar  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DPLX = ROOT / "data/lensmodel/bestopt_1.524_dplx.fits"
DPLY = ROOT / "data/lensmodel/bestopt_1.524_dply.fits"
QSO_1A = (178.3352306, 7.9335254)  # PSF-fitted JWST position of 1A (M6b), RA, Dec (deg)
OBS = {"1A": (178.3352456, 7.9335292), "1B": (178.3317916, 7.9364979), "1C": (178.3289710, 7.9369786)}


# --- the picture ----------------------------------------------------------------


def load_picture(path, max_rows=400):
    """RGBA picture -> (rows, cols, 4) float array, colours premultiplied by alpha.

    Cropped to its non-transparent pixels and shrunk to at most ``max_rows``
    rows (smooth resampling: fewer source pixels per image pixel, less
    aliasing). Row 0 is the TOP of the picture. Premultiplying (r a, g a, b a, a)
    makes interpolation and averaging blend edges correctly on black.
    """
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3]
    r, c = np.nonzero(a > 0)
    im = im.crop((c.min(), r.min(), c.max() + 1, r.max() + 1))
    if im.height > max_rows:
        im = im.resize((round(im.width * max_rows / im.height), max_rows), Image.LANCZOS)
    x = np.asarray(im, float) / 255.0
    x[..., :3] *= x[..., 3:4]
    return x


def sample_picture(pic, bx, by, centre, height):
    """The picture's premultiplied RGBA at source positions (bx, by) (arcsec).

    pic : (rows, cols, 4) from load_picture
    centre : (beta_x, beta_y) of the picture's middle; height : its height (arcsec)
    Outside the picture: 0 (transparent). The picture is upright with east
    left: column index grows towards -x (west), row index towards -y (south).
    Returns an array of shape bx.shape + (4,).
    """
    nr, nc = pic.shape[:2]
    s = height / nr  # arcsec per picture pixel
    col = -(bx - centre[0]) / s + (nc - 1) / 2
    row = -(by - centre[1]) / s + (nr - 1) / 2
    coords = np.stack([row.ravel(), col.ravel()])
    out = [map_coordinates(pic[..., k], coords, order=1, mode="constant", cval=0.0) for k in range(4)]
    return np.stack(out, axis=-1).reshape(bx.shape + (4,))


# --- grids ----------------------------------------------------------------------


def display_grid(xlim, ylim, n, ss):
    """Supersampled grid of an n x n panel: x from xlim[0] (left, east) to xlim[1].

    Returns (x, y) arrays of shape (n*ss, n*ss); row 0 = ylim[0] (bottom, for
    imshow origin="lower"), column 0 = xlim[0].
    """
    m = n * ss
    dx = (xlim[1] - xlim[0]) / m
    dy = (ylim[1] - ylim[0]) / m
    x = xlim[0] + (np.arange(m) + 0.5) * dx
    y = ylim[0] + (np.arange(m) + 0.5) * dy
    return np.meshgrid(x, y)


def image_plane_limits(field, fov):
    """(xlim, ylim) of the image-plane panel (arcsec): a fov x fov square
    centred on the mean of the observed quasar images; xlim[0] > xlim[1]
    (east, +x, on the left)."""
    th = [[float(v) for v in field.sky_to_theta(*rd)] for rd in OBS.values()]
    cx, cy = np.mean(th, axis=0)
    h = fov / 2
    return (cx + h, cx - h), (cy - h, cy + h)


def bin_down(img, ss):
    """Average ss x ss blocks: (m, m, 4) -> (m/ss, m/ss, 4)."""
    m = img.shape[0] // ss
    return img.reshape(m, ss, m, ss, -1).mean(axis=(1, 3))


def backtrace_chunked(field, tx, ty, chunk=250_000):
    """beta(theta) for large grids, in chunks (bounded memory)."""
    bx, by = np.empty(tx.size), np.empty(tx.size)
    fx, fy = tx.ravel(), ty.ravel()
    for i in range(0, tx.size, chunk):
        a, b = field.backtrace(fx[i:i + chunk], fy[i:i + chunk])
        bx[i:i + chunk], by[i:i + chunk] = np.asarray(a), np.asarray(b)
    return bx.reshape(tx.shape), by.reshape(ty.shape)


def smoothstep(t):
    return t * t * (3 - 2 * t)


# --- main -----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=str(HERE / "bernie.png"))
    ap.add_argument("--height", type=float, default=2.0, help="picture height in the source plane (arcsec)")
    ap.add_argument("--start", type=float, nargs=2, default=None,
                    help="start offset from the quasar's beta (arcsec, east north); default 5.5 4.5")
    ap.add_argument("--frames", type=int, default=90, help="frames for the move (plus holds at both ends)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--pixels", type=int, default=600, help="pixels per panel side")
    ap.add_argument("--ss", type=int, default=3, help="supersampling per pixel side")
    ap.add_argument("--fov", type=float, default=44.0, help="image-plane field of view (arcsec)")
    ap.add_argument("--clean", action="store_true",
                    help="pictures only: no axes, titles, caustics, critical curves or quasar markers")
    ap.add_argument("--same-scale", action="store_true",
                    help="source panel with the image panel's field of view (1-to-1 angular sizes; on with --clean)")
    ap.add_argument("--image-only", action="store_true",
                    help="only the image plane, pictures only (implies --clean)")
    ap.add_argument("--out", default=None, help="default: bernie_lensing.gif, bernie_lensing_clean.gif with --clean, "
                                                "bernie_lensing_image.gif with --image-only")
    a = ap.parse_args(argv)
    a.clean = a.clean or a.image_only
    a.same_scale = a.same_scale or a.clean
    if a.out is None:
        name = "image" if a.image_only else "clean" if a.clean else None
        a.out = str(HERE / (f"bernie_lensing_{name}.gif" if name else "bernie_lensing.gif"))
    t0 = time.perf_counter()

    field = load_deflection(DPLX, DPLY, z_ref=1.524)
    beta_q = np.asarray(image_fixed_quasar(*QSO_1A, field).beta, float)
    start = beta_q + np.asarray(a.start if a.start is not None else (5.5, 4.5))
    cc = critical_curves(field)
    ca = caustics(field, cc)
    pic = load_picture(a.image)
    print(f"quasar beta = ({beta_q[0]:+.3f}, {beta_q[1]:+.3f})\"; picture {pic.shape[1]} x {pic.shape[0]} px, "
          f"{a.height}\" tall; start ({start[0]:+.2f}, {start[1]:+.2f})\"", flush=True)

    # Image plane: centred on the three quasar images. Source plane: the
    # path plus the main caustic around it.
    img_xlim, img_ylim = image_plane_limits(field, a.fov)
    h = a.fov / 2
    sx, sy = (start[0] + beta_q[0]) / 2, (start[1] + beta_q[1]) / 2
    hs = max(abs(start - beta_q)) / 2 + 3.0 * a.height
    if a.same_scale:  # same arcsec per pixel in both panels: sizes compare 1-to-1
        hs = h
    src_xlim, src_ylim = (sx + hs, sx - hs), (sy - hs, sy + hs)

    tx, ty = display_grid(img_xlim, img_ylim, a.pixels, a.ss)
    bx, by = backtrace_chunked(field, tx, ty)  # the lens, evaluated once
    sgx, sgy = display_grid(src_xlim, src_ylim, a.pixels, a.ss)
    print(f"traced {tx.size} image-plane points ({time.perf_counter() - t0:.0f} s)", flush=True)

    plt.style.use("dark_background")
    if a.image_only:  # only the image plane, nothing but the picture
        fig = plt.figure(figsize=(6, 6))
        axi = fig.add_axes((0, 0, 1, 1))
        axs = fig.add_axes((0, 0, 0.01, 0.01))  # unused, invisible
        axs.set_visible(False)
    elif a.clean:  # only the two pictures, side by side
        fig, (axs, axi) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)  # each panel exactly pixels x pixels
    else:
        fig, (axs, axi) = plt.subplots(1, 2, figsize=(12, 6.3), layout="constrained")
    empty = np.zeros((a.pixels, a.pixels, 3))
    ims = axs.imshow(empty, origin="lower", extent=(*src_xlim, *src_ylim), interpolation="nearest")
    imi = axi.imshow(empty, origin="lower", extent=(*img_xlim, *img_ylim), interpolation="nearest")
    for ax, lim in ((axs, (src_xlim, src_ylim)), (axi, (img_xlim, img_ylim))):
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
        ax.set_aspect("equal")
    if a.clean:
        axs.set_axis_off()
        axi.set_axis_off()
    else:
        for k, c in enumerate(ca):
            axs.plot(c[:, 0], c[:, 1], "-", color="c", lw=0.7, alpha=0.8, label="caustic" if k == 0 else None)
        for k, c in enumerate(cc):
            axi.plot(c[:, 0], c[:, 1], "-", color="red", lw=0.7, alpha=0.8,
                     label="critical curve" if k == 0 else None)
        axs.set_title("source plane", fontsize=13)
        axi.set_title("image plane", fontsize=13)
        handles = axs.get_legend_handles_labels()[0] + axi.get_legend_handles_labels()[0]
        axi.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.6)
        axs.set_xlabel(r"$\beta_x$ [arcsec]  (east $\leftarrow$)")
        axs.set_ylabel(r"$\beta_y$ [arcsec]  (north $\uparrow$)")
        axi.set_xlabel(r"$\theta_x$ [arcsec]  (east $\leftarrow$)")
        axi.set_ylabel(r"$\theta_y$ [arcsec]")

    hold0, hold1 = 15, 40
    ts = [0.0] * hold0 + list(smoothstep(np.linspace(0, 1, a.frames))) + [1.0] * hold1
    writer = PillowWriter(fps=a.fps)
    with writer.saving(fig, a.out, dpi=100):
        last = None
        for i, t in enumerate(ts):
            centre = (1 - t) * start + t * beta_q
            if last is None or t != last:  # held frames reuse the rendering
                src = bin_down(sample_picture(pic, sgx, sgy, centre, a.height), a.ss)
                lensed = bin_down(sample_picture(pic, bx, by, centre, a.height), a.ss)
                ims.set_data(np.clip(src[..., :3], 0, 1))  # premultiplied colour = composite on black
                imi.set_data(np.clip(lensed[..., :3], 0, 1))
                last = t
            writer.grab_frame()
            if i % 20 == 0:
                print(f"  frame {i + 1}/{len(ts)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB, {len(ts)} frames, "
          f"{time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
