"""A still PNG of the foreground cluster members, to lay over the image plane of
bernie_lensing_image.gif / the image-plane half of bernie_lensing_clean.gif.

The JWST colour image (red F444W, green F150W, blue F115W; all on the same
0.03" mosaic grid) is resampled onto exactly the image-plane panel of
bernie_lensing.py (same field of view, north up, east left, same pixels).
Only sources that contain a cluster member are kept; everything else
(sky, quasar images, arcs, background galaxies) is transparent (alpha = 0).

Cluster members: the Lenstool model's member catalogue (cats/1153ground.cat:
id, RA, Dec, ...) plus the individually optimized galaxies O2-O5 of
bestopt.par (Lenstool offsets from the reference position, x = west, so
theta_x = -x). Only positions are read from the model.

Run from the repository root:
    python demos/bernie_lensing/cluster_members.py [--fov 44] [--pixels 600] [--check]
--check also writes cluster_members_check.png (full colour image with the
kept sources outlined and the member positions marked).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from PIL import Image
from scipy import ndimage

import forwardgal  # noqa: F401
from forwardgal.lens import load_deflection

from bernie_lensing import DPLX, DPLY, HERE, OBS, ROOT, bin_down, display_grid, image_plane_limits

MODEL = Path.home() / "Research/Tools/lenstool_tools/single_model/dec19a_model_solhaug"
BANDS = {  # colour channel -> mosaic (SCI extension, MJy/sr)
    "r": ROOT / "data/jw06675-o001_t002_nircam_clear-f444w_i2d.fits",
    "g": ROOT / "data/jw06675-o001_t002_nircam_clear-f150w_i2d.fits",
    "b": Path.home() / "Research/Data_and_Analysis/DATA/CJ1153/jw06675-o001_t002_nircam_clear-f115w_i2d.fits",
}
O_GALAXIES = {"O2": (4.693, -18.695), "O3": (-0.516, 14.783), "O4": (-10.089, 12.483), "O5": (-5.659, 15.838)}


def members_theta(field):
    """(N, 2) theta (arcsec) of the cluster members (catalogue + O2-O5)."""
    cat = np.loadtxt(MODEL / "cats/1153ground.cat", usecols=(1, 2))
    tx, ty = field.sky_to_theta(cat[:, 0], cat[:, 1])
    th = np.column_stack([np.asarray(tx, float), np.asarray(ty, float)])
    lenstool = np.array(list(O_GALAXIES.values()))
    return np.vstack([th, np.column_stack([-lenstool[:, 0], lenstool[:, 1]])])


def resample_band(path, ra, dec):
    """Mosaic values at (ra, dec) (deg, any shape), bilinear; reads only the needed box."""
    with fits.open(path, memmap=True) as hdul:
        sci = hdul["SCI"]
        col, row = WCS(sci.header).celestial.world_to_pixel_values(ra, dec)  # 0-based
        r0, r1 = int(np.floor(row.min())) - 2, int(np.ceil(row.max())) + 3
        c0, c1 = int(np.floor(col.min())) - 2, int(np.ceil(col.max())) + 3
        box = np.nan_to_num(np.asarray(sci.section[r0:r1, c0:c1], float))
    return ndimage.map_coordinates(box, [row - r0, col - c0], order=1, mode="constant", cval=0.0)


def to_pixels(theta, xlim, ylim, n):
    """theta (N, 2) -> panel (row, col) (0-based; row 0 at the bottom, as display_grid)."""
    col = (theta[:, 0] - xlim[0]) / ((xlim[1] - xlim[0]) / n) - 0.5
    row = (theta[:, 1] - ylim[0]) / ((ylim[1] - ylim[0]) / n) - 0.5
    return row, col


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fov", type=float, default=44.0, help="must match bernie_lensing.py --fov")
    ap.add_argument("--pixels", type=int, default=600, help="must match bernie_lensing.py --pixels")
    ap.add_argument("--ss", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=3.0, help="detection threshold (sky sigma)")
    ap.add_argument("--grow", type=int, default=3, help="dilate the kept sources by this many pixels")
    ap.add_argument("--red-cut", type=float, default=0.4,
                    help="pixels redder than this (2.5 log10 F444W/blue) are lensed light, not members")
    ap.add_argument("--alpha-full", type=float, default=0.25,
                    help="brightness (0-1 display units) above which a pixel is fully opaque")
    ap.add_argument("--alpha-floor", type=float, default=0.03,
                    help="brightness above the sky below which a pixel is fully transparent")
    ap.add_argument("--red-grow", type=int, default=4, help="grow the red (lensed) pixels by this many pixels")
    ap.add_argument("--out", default=str(HERE / "cluster_members.png"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    field = load_deflection(DPLX, DPLY, z_ref=1.524)
    xlim, ylim = image_plane_limits(field, a.fov)
    tx, ty = display_grid(xlim, ylim, a.pixels, a.ss)
    ra, dec = (np.asarray(v, float) for v in field.theta_to_sky(tx, ty))
    bands = {k: bin_down(resample_band(p, ra, dec)[..., None], a.ss)[..., 0] for k, p in BANDS.items()}

    # Detection: each band in units of its sky noise (robust: MAD), summed, smoothed.
    det = 0
    for img in bands.values():
        med = np.median(img)
        det = det + (img - med) / (1.4826 * np.median(np.abs(img - med)))
    det = ndimage.gaussian_filter(det / np.sqrt(len(bands)), 1.0)
    # Lensed z = 1.5 light (quasar images, host arcs) is much redder than the
    # z = 0.43 members: colour = 2.5 log10(F444W / mean(F150W, F115W)) of the
    # smoothed images is +1.1..+1.3 on the lensed features and -0.6..-0.1 at
    # the members (measured). Remove red pixels (grown over the F444W PSF
    # wings) so lensed light touching a member is not kept with it.
    sm = {k: ndimage.gaussian_filter(v - np.median(v), 1.5) for k, v in bands.items()}
    colour = 2.5 * np.log10(np.clip(sm["r"], 1e-4, None) / np.clip(0.5 * (sm["g"] + sm["b"]), 1e-4, None))
    red = ndimage.binary_dilation((colour > a.red_cut) & (det > a.threshold), iterations=a.red_grow)
    labels, n_src = ndimage.label((det > a.threshold) & ~red)
    mr, mc = to_pixels(members_theta(field), xlim, ylim, a.pixels)
    inside = (mr >= 0) & (mr < a.pixels) & (mc >= 0) & (mc < a.pixels)
    keep_ids = set(labels[np.round(mr[inside]).astype(int), np.round(mc[inside]).astype(int)]) - {0}
    keep = ndimage.binary_dilation(np.isin(labels, list(keep_ids)), iterations=a.grow) & ~red
    print(f"{n_src} sources detected; {inside.sum()} member positions in the field; {len(keep_ids)} sources kept")

    # Colour (Lupton asinh) and alpha: opaque on the galaxies, fading out with
    # their light, zero elsewhere.
    scale = {k: 1.0 / np.percentile(v[keep], 99.0) for k, v in bands.items()}
    rgb = make_lupton_rgb(*(np.clip(bands[k] * scale[k], 0, None) for k in "rgb"), minimum=0.0, stretch=0.05,
                          Q=8) / 255.0
    # Alpha from the smoothed brightness (a soft ramp above the sky), times a
    # feathered version of the kept-source mask: galaxies fade out smoothly.
    lum = ndimage.gaussian_filter(rgb.max(axis=-1), 1.5)
    sky = np.median(lum[~keep]) + a.alpha_floor
    alpha = np.clip((lum - sky) / (a.alpha_full - sky), 0, 1) * np.clip(
        ndimage.gaussian_filter(keep.astype(float), 1.5) * 2 - 0.5, 0, 1)
    # The faint outskirts carry coloured pixel noise: take the hue from a
    # strongly smoothed image and the brightness from a lightly smoothed one.
    smooth = np.stack([ndimage.gaussian_filter(rgb[..., k], 3.0) for k in range(3)], axis=-1)
    hue = smooth / np.maximum(smooth.max(axis=-1, keepdims=True), 1e-6)
    bright = ndimage.gaussian_filter(rgb.max(axis=-1), 0.8)[..., None]
    shown = hue * bright  # what the PNG should look like over black
    # Store straight (un-premultiplied) colour so that colour x alpha = shown.
    colour = np.clip(shown / np.maximum(alpha, 1e-3)[..., None], 0, 1)
    out = np.concatenate([colour, alpha[..., None]], axis=-1)[::-1]  # row 0 = top (north) in a PNG
    Image.fromarray((out * 255).round().astype(np.uint8), "RGBA").save(a.out)
    print(f"wrote {a.out} ({a.pixels} x {a.pixels})")

    if a.check:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(rgb, origin="lower")
        ax.contour(keep, levels=[0.5], colors="lime", linewidths=0.7)
        ax.plot(mc[inside], mr[inside], "x", color="yellow", ms=6)
        q = np.array([[float(v) for v in field.sky_to_theta(*rd)] for rd in OBS.values()])
        qr, qc = to_pixels(q, xlim, ylim, a.pixels)
        ax.plot(qc, qr, "o", mfc="none", color="red", ms=12)
        ax.set_title("kept sources (green), members (yellow x), quasar images (red o)")
        fig.savefig(Path(a.out).with_name("cluster_members_check.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
