"""Which critical curves (and masses) make the caustics in the masked source region?

Every caustic is the image of one critical curve under beta = theta - alpha.
This finds the caustics that overlap the source-plane region constrained by
the mask (the mask traced to the source plane), gives each a colour and an
ID, and shows:

  left : the image plane (wide view): data, the mask, quasar images 1A/1B/1C,
         all critical curves (grey) and the relevant ones (coloured, labelled),
         Lenstool's individually modelled potentials O1-O6 (x) and potfile
         cluster members (o);
  right: the source plane: traced mask outline, all caustics (grey), the
         relevant ones (coloured, labelled) and the quasar.

A caustic crossing the traced mask means: some source points that the mask
constrains have EXTRA images near that caustic's critical curve, wherever in
the image plane that curve is.

Run from the repository root:
    python scripts/m8c_caustic_origins.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from scipy.ndimage import binary_closing

import forwardgal  # noqa: F401
from common import IMAGE, LENSTOOL_MODEL, QSO_1A, example_mask, load_example_cutout, load_field
from forwardgal.critical import caustics, critical_curves
from forwardgal.data import load_cutout
from forwardgal.quasar import image_fixed_quasar
from forwardgal.regions import fit_region
from forwardgal.traced import trace_grid

field = load_field()
qso = image_fixed_quasar(*QSO_1A, field)
crit = critical_curves(field)
caus = caustics(field, crit)

# The mask traced to the source plane, as a filled region on a fine beta grid.
img = load_example_cutout()
region = fit_region(example_mask(img.data.shape), margin=35)
r0, r1, c0, c1 = region.box
grid = trace_grid(img.wcs, region.box_shape, field, subsample=1, origin=(r0, c0))
mask = region.in_box(region.mask)
bmx, bmy = grid.beta_sub_x[..., 0][mask], grid.beta_sub_y[..., 0][mask]
size, step = 2.4, 0.005
ex = np.arange(qso.beta[0] - size / 2, qso.beta[0] + size / 2 + step, step)
ey = np.arange(qso.beta[1] - size / 2, qso.beta[1] + size / 2 + step, step)
covered = binary_closing(np.histogram2d(bmy, bmx, bins=[ey, ex])[0] > 0, iterations=3)


def in_covered(pts):
    i = np.searchsorted(ey, pts[:, 1]) - 1
    j = np.searchsorted(ex, pts[:, 0]) - 1
    ok = (i >= 0) & (i < covered.shape[0]) & (j >= 0) & (j < covered.shape[1])
    out = np.zeros(len(pts), bool)
    out[ok] = covered[i[ok], j[ok]]
    return out


relevant = [k for k, c in enumerate(caus) if in_covered(c).any()]

# Mass components: Lenstool potentiels (x is WEST-positive in Lenstool) and potfile members.
pots = {}
name = None
for line in (LENSTOOL_MODEL / "bestopt_1.524analysis.par").read_text().splitlines():
    t = line.split()
    if line.startswith("potentiel"):
        name, pots[t[1]] = t[1], {}
    elif name and t and t[0] in ("x_centre", "y_centre", "v_disp"):
        pots[name][t[0]] = float(t[1])
    elif line.startswith("end") or (line and not line[0].isspace()):
        name = None if not line.startswith("potentiel") else name
pot_theta = {k: (-v["x_centre"], v["y_centre"], v["v_disp"]) for k, v in pots.items() if "x_centre" in v}
# Potfile catalogue: "id ra dec a b theta mag lum". IDs may be text (e.g. "A7");
# lines starting with "#" are commented out (ignored by Lenstool), typically
# galaxies modelled individually as a potentiel instead.
mem_id, mem_active, mem_ra, mem_dec, mem_mag = [], [], [], [], []
for line in (LENSTOOL_MODEL / "cats/1153ground.cat").read_text().splitlines():
    t = line.split()
    if len(t) >= 7:
        mem_id.append(t[0].lstrip("#"))
        mem_active.append(not t[0].startswith("#"))
        mem_ra.append(float(t[1]))
        mem_dec.append(float(t[2]))
        mem_mag.append(float(t[6]))
mem_active, mem_mag = np.array(mem_active), np.array(mem_mag)
mtx, mty = (np.asarray(v) for v in field.sky_to_theta(np.array(mem_ra), np.array(mem_dec)))

print(f"caustics overlapping the traced mask: {len(relevant)} of {len(caus)}")
colours = plt.cm.tab10(np.arange(10))[[0, 2, 3, 4, 5, 8, 9]]  # blue, green, red, ... (not the marker colours)
labels = {}
for n, k in enumerate(relevant):
    c = crit[k]
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    dp = {p: np.hypot(v[0] - cx, v[1] - cy) for p, v in pot_theta.items()}
    p_near = min(dp, key=dp.get)
    dm = np.hypot(mtx - cx, mty - cy)
    m_near = int(np.argmin(dm))
    who = (f"galaxy {mem_id[m_near]} (mag {mem_mag[m_near]:.1f}, "
           f"{'in the potfile' if mem_active[m_near] else 'commented out of the potfile'}), {dm[m_near]:.2f}\" away")
    if dp[p_near] < 0.5:
        who += f"; modelled as potentiel {p_near} (sigma {pot_theta[p_near][2]:.0f} km/s), {dp[p_near]:.2f}\" away"
    big = np.ptp(c[:, 0]) > 10
    labels[k] = f"C{n + 1}"
    print(f"  {labels[k]}: critical curve centre theta = ({cx:+6.2f}, {cy:+6.2f})\", size {np.ptp(c[:, 0]):5.2f} x {np.ptp(c[:, 1]):5.2f}\""
          f"{' (cluster-scale)' if big else ''}; nearest mass: {who}")

# Wide image-plane background: data around the three quasar images.
ra_c, dec_c = field.theta_to_sky(7.0, 10.0)
wide = load_cutout(IMAGE, "SCI", IMAGE, "ERR", float(ra_c), float(dec_c), size_arcsec=32.0)
wg = trace_grid(wide.wcs, (2, 2), field, subsample=1)  # only for its theta corner extent
ny, nx = wide.data.shape
corners = [field.sky_to_theta(*wide.wcs.wcs_pix2world(c_, r_, 0)) for r_, c_ in [(-0.5, -0.5), (ny - 0.5, nx - 0.5)]]
ext_w = (float(corners[0][0]), float(corners[1][0]), float(corners[0][1]), float(corners[1][1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.imshow(wide.data, origin="lower", extent=ext_w, cmap="gray",
           norm=ImageNormalize(wide.data, vmin=-0.05, vmax=1.0, stretch=AsinhStretch(0.05)))
for k, c in enumerate(crit):
    if k not in relevant:
        ax1.plot(c[:, 0], c[:, 1], "-", color="0.55", lw=0.7)
for n, k in enumerate(relevant):
    c = crit[k]
    ax1.plot(c[:, 0], c[:, 1], "-", color=colours[n % len(colours)], lw=1.8)
    ax1.annotate(labels[k], (c[:, 0].max(), c[:, 1].max()), color=colours[n % len(colours)], fontsize=10, weight="bold")
tmx, tmy = grid.theta_corner_x[:-1, :-1][mask], grid.theta_corner_y[:-1, :-1][mask]
ax1.plot(tmx, tmy, ",", color="c", alpha=0.25)
ax1.plot([], [], "s", color="c", ms=5, label="mask")
for nm, (ra, dec) in {"1A": QSO_1A, "1B": (178.3317916, 7.9364979), "1C": (178.3289710, 7.9369786)}.items():
    t = [float(v) for v in field.sky_to_theta(ra, dec)]
    ax1.plot(*t, "*", color="lime", ms=12)
    ax1.annotate(nm, t, xytext=(5, -12), textcoords="offset points", color="lime", fontsize=10)
for p, (x, y, s) in pot_theta.items():
    if ext_w[1] < x < ext_w[0] and ext_w[2] < y < ext_w[3]:
        ax1.plot(x, y, "x", color="yellow", ms=10, mew=2)
        ax1.annotate(p, (x, y), xytext=(5, 5), textcoords="offset points", color="yellow", fontsize=9)
inside = (mtx < ext_w[0]) & (mtx > ext_w[1]) & (mty > ext_w[2]) & (mty < ext_w[3])
ax1.plot(mtx[inside & mem_active], mty[inside & mem_active], "o", mfc="none", mec="magenta", ms=7, mew=1.2,
         label="cluster members (potfile)")
ax1.plot(mtx[inside & ~mem_active], mty[inside & ~mem_active], "s", mfc="none", mec="magenta", ms=8, mew=1.2,
         label="commented out of the potfile")
ax1.plot([], [], "x", color="yellow", label="potentiel O1-O6")
ax1.set_xlim(ext_w[0], ext_w[1])
ax1.set_ylim(ext_w[2], ext_w[3])
ax1.set_xlabel("theta_x [arcsec, +east]")
ax1.set_ylabel("theta_y [arcsec, +north]")
ax1.set_title("image plane: critical curves whose caustics cross the traced mask (coloured)", fontsize=10)
ax1.legend(loc="lower left", fontsize=8)

ax2.contour(covered, levels=[0.5], colors="c", linewidths=1.5,
            extent=(ex[0], ex[-1], ey[0], ey[-1]), origin="lower")
ax2.plot([], [], "c-", label="mask, traced to the source plane")
for k, c in enumerate(caus):
    if k not in relevant:
        ax2.plot(c[:, 0], c[:, 1], "-", color="0.6", lw=0.7)
for n, k in enumerate(relevant):
    c = caus[k]
    ax2.plot(c[:, 0], c[:, 1], "-", color=colours[n % len(colours)], lw=1.8)
    j = np.argmax(c[:, 1])
    ax2.annotate(labels[k], (c[j, 0], c[j, 1]), color=colours[n % len(colours)], fontsize=10, weight="bold")
ax2.plot(*qso.beta, "*", color="lime", ms=14, label="quasar")
ax2.set_xlim(qso.beta[0] + size / 2, qso.beta[0] - size / 2)
ax2.set_ylim(qso.beta[1] - size / 2, qso.beta[1] + size / 2)
ax2.set_aspect("equal")
ax2.set_xlabel("beta_x [arcsec, +east]")
ax2.set_ylabel("beta_y [arcsec, +north]")
ax2.set_title("source plane: the matching caustics", fontsize=10)
ax2.legend(loc="lower left", fontsize=8)
out = Path("outputs/m8c_caustic_origins.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved figure: {out}")
plt.show()
