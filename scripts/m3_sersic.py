"""Milestone 3: render an unlensed Sersic and check its total flux analytically.

Prints, for several Sersic indices n, effective radii (in pixels) and
subsampling factors s, the relative error of the rendered flux
(sum of pixel values x pixel area) against the analytic total flux L.

Run from the repository root:
    python scripts/m3_sersic.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import forwardgal  # noqa: F401  (enables 64-bit JAX)
from forwardgal.grids import bin_mean, local_theta, subpixel_centres
from forwardgal.profiles.sersic import ie_to_mag, sersic, sersic_total_flux

PIXSCALE = 0.03  # arcsec/px (NIRCam F115W i2d)
NPIX = 401  # grid side; centre at pixel 200
ZP = 28.087  # AB zeropoint of the F115W mosaic (M1)
Q, PA = 0.6, 30.0


def render(n, R_e, s, I_e=1.0):
    rows, cols = subpixel_centres(NPIX, NPIX, s)
    tx, ty = local_theta(rows, cols, PIXSCALE, NPIX // 2, NPIX // 2)
    return bin_mean(sersic(tx, ty, 0.0, 0.0, I_e, R_e, n, Q, PA), s)


print("relative flux error  (rendered - analytic) / analytic")
print(f"{'n':>4} {'R_e[px]':>8} " + " ".join(f"{'s=' + str(s):>10}" for s in (1, 3, 5, 9)))
for n in (0.5, 1.0, 4.0):
    for re_pix in (1.0, 3.0, 10.0):
        R_e = re_pix * PIXSCALE
        L = float(sersic_total_flux(1.0, R_e, n, Q))
        errs = []
        for s in (1, 3, 5, 9):
            numeric = float(render(n, R_e, s).sum()) * PIXSCALE**2
            errs.append(numeric / L - 1)
        print(f"{n:4.1f} {re_pix:8.1f} " + " ".join(f"{e:+10.2e}" for e in errs))

t0 = time.perf_counter()
img = render(1.0, 3 * PIXSCALE, 5)
print(f"\nrender time (401x401, s=5) : {1e3 * (time.perf_counter() - t0):.0f} ms")
print(f"m_tot for I_e = 1 MJy/sr, R_e = 0.09\", n = 1: {float(ie_to_mag(1.0, 0.09, 1.0, Q, ZP, PIXSCALE)):.3f} AB")

# Picture: east left, north up; the major axis should point 30 deg east of north.
half = NPIX // 2 * PIXSCALE
fig, ax = plt.subplots(figsize=(5.5, 5))
crop = img[NPIX // 2 - 30 : NPIX // 2 + 31, NPIX // 2 - 30 : NPIX // 2 + 31]
ext = 30 * PIXSCALE
ax.imshow(np.log10(crop), origin="lower", extent=(ext, -ext, -ext, ext), cmap="magma")
ax.plot([0, 0.6 * np.sin(np.radians(PA))], [0, 0.6 * np.cos(np.radians(PA))], "c-", label="PA = 30° E of N")
ax.set_xlim(ext, -ext)
ax.set_xlabel("theta_x [arcsec, +east]")
ax.set_ylabel("theta_y [arcsec, +north]")
ax.set_title("log10 I, n = 1, q = 0.6")
ax.legend(loc="lower right")
fig.savefig("outputs/m3_sersic.png", dpi=110, bbox_inches="tight")
print("saved figure      : outputs/m3_sersic.png")
plt.show()
