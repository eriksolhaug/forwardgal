"""Milestone 2a: load the Lenstool deflection maps and locate the quasar images.

Run from the repository root:
    python scripts/m2a_deflection.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forwardgal.lens import load_deflection

field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits",
    "data/lensmodel/bestopt_1.524_dply.fits",
    z_ref=1.524,
)

# Quasar images, from arcs/1153arc_v2_zvary.dat (system 1, z = 1.524).
QSO = {
    "1A": (178.3352456, 7.9335292),
    "1B": (178.3317916, 7.9364979),
    "1C": (178.3289710, 7.9369786),
}

ny, nx = field.alpha_x.shape
print(f"map shape        : {field.alpha_x.shape}  ([row, col])")
print(f"step_x, step_y   : {field.step_x:+.5f}, {field.step_y:+.5f} arcsec/px")
print(f"reference pixel  : [row, col] = [{field.i0:.0f}, {field.j0:.0f}] (0-based)")
corner_x, corner_y = field.pixel_to_theta(np.array([0, ny - 1]), np.array([0, nx - 1]))
print(f"theta_x range    : {corner_x[0]:+.2f} .. {corner_x[1]:+.2f} arcsec (col 0 .. {nx - 1})")
print(f"theta_y range    : {corner_y[0]:+.2f} .. {corner_y[1]:+.2f} arcsec (row 0 .. {ny - 1})")
print()
print("image   theta_x   theta_y   [row, col] on map")
for name, (ra, dec) in QSO.items():
    tx, ty = field.sky_to_theta(ra, dec)
    row, col = field.theta_to_pixel(tx, ty)
    print(f"{name:5s} {tx:+8.3f}″ {ty:+8.3f}″   [{row:6.1f}, {col:6.1f}]")

# Plot the maps in theta coordinates. extent = (left, right, bottom, top) in
# theta; left is theta_x at col 0 (positive, east), so east ends up on the left.
half = 0.5 * np.array([field.step_x, field.step_y])
extent = (corner_x[0] - half[0], corner_x[1] + half[0], corner_y[0] - half[1], corner_y[1] + half[1])
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, arr, name in zip(axes, [field.alpha_x, field.alpha_y], ["alpha_x (east)", "alpha_y (north)"]):
    im = ax.imshow(arr, origin="lower", extent=extent, cmap="RdBu_r", vmin=-25, vmax=25)
    for qname, (ra, dec) in QSO.items():
        tx, ty = field.sky_to_theta(ra, dec)
        ax.plot(tx, ty, "k+", ms=10)
        ax.annotate(qname, (tx, ty), xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(30, -30)  # east (positive theta_x) on the left
    ax.set_ylim(-30, 30)
    ax.set_xlabel("theta_x [arcsec, +east]")
    ax.set_ylabel("theta_y [arcsec, +north]")
    ax.set_title(f"{name} [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046)

out = Path("outputs/m2a_deflection.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved figure     : {out}")
plt.show()
