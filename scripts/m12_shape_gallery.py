"""Milestone 12: what Fourier modes and bending modes do (forwardgal vs GALFIT).

One Sersic (n = 1, R_e = 40 px, q = 0.4, PA = 30 deg, north up, east left)
with F1, F3, F4, B2, B3 and a combination. Row 1: forwardgal (point values,
log scale, isophotes); row 2: GALFIT 3 model-only images, if GALFIT is found
(FORWARDGAL_GALFIT); row 3: their difference / peak. Bending amplitudes are
given here in GALFIT pixels (forwardgal: arcsec = pixels x 0.1"/px), with
GALFIT's convention (pa_ref = 0).
Writes outputs/m12_shape_gallery.png.

Run from the repository root:
    FORWARDGAL_GALFIT=~/Research/Tools/galfit/galfit python scripts/m12_shape_gallery.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tests")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import forwardgal  # noqa: E402,F401
from forwardgal.profiles.sersic import Bending, sersic  # noqa: E402
from test_galfit import CENTRE, GALFIT, N, PIXSCALE, galfit_bent  # noqa: E402

RE, NS, Q, PA = 40.0, 1.0, 0.4, 30.0
CASES = [  # (title, Fourier modes (m, a, phi) or None, bends {m: pixels})
    ("plain", None, {}),
    ("F1: a=0.15, phi=0 (lopsided)", ((1,), (0.15,), (0.0,)), {}),
    ("F3: a=0.08, phi=20 (triangular)", ((3,), (0.08,), (20.0,)), {}),
    ("F4: a=0.08, phi=0 (boxy/discy)", ((4,), (0.08,), (0.0,)), {}),
    ("B2: 6 px (banana)", None, {2: 6.0}),
    ("B3: 4 px (S-shape)", None, {3: 4.0}),
    ("F1 + B2 + B3", ((1,), (0.1,), (40.0,)), {2: 5.0, 3: -2.0}),
]

rows, cols = np.indices((N, N))
c = CENTRE - 1
x, y = -(cols - c) * PIXSCALE, (rows - c) * PIXSCALE
s = slice(c - 110, c + 111)
fig, axes = plt.subplots(3 if GALFIT else 1, len(CASES), figsize=(3.0 * len(CASES), 9.3 if GALFIT else 3.4),
                         squeeze=False)
for k, (title, modes, bends) in enumerate(CASES):
    bending = Bending(tuple(bends), [a * PIXSCALE for a in bends.values()], 0.0) if bends else None
    ours = np.array(sersic(x, y, 0.0, 0.0, 1.0, RE * PIXSCALE, NS, Q, PA, modes=modes, bending=bending))
    ours /= ours.max()
    panels = [ours]
    if GALFIT:
        with tempfile.TemporaryDirectory() as d:
            g = galfit_bent(Path(d), RE, NS, Q, PA, bends, modes)
        ok = np.hypot(rows - c, cols - c) > 15
        kk = (g[ok] * ours[ok]).sum() / (ours[ok] ** 2).sum()
        diff = np.where(ok, g - kk * ours, np.nan) / g.max()
        panels += [g / g.max(), diff]
        print(f"{title:32s} max |GALFIT - forwardgal| / peak (r > 15 px) = {np.nanmax(np.abs(diff)):.1e}")
    for r, img in enumerate(panels):
        ax = axes[r, k]
        if r < 2:
            ax.imshow(np.log10(np.maximum(img[s, s], 1e-4)), origin="lower", cmap="magma", vmin=-3, vmax=0)
            ax.contour(np.log10(np.maximum(img[s, s], 1e-4)), levels=[-2.5, -2, -1.5, -1, -0.5], colors="w",
                       linewidths=0.5, linestyles="solid")
        else:
            ax.imshow(img[s, s], origin="lower", cmap="RdBu_r", vmin=-1e-4, vmax=1e-4)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, k].set_title(title, fontsize=9)
axes[0, 0].set_ylabel("forwardgal")
if GALFIT:
    axes[1, 0].set_ylabel("GALFIT 3.0.5")
    axes[2, 0].set_ylabel("difference / peak\n(+-1e-4, r > 15 px)")
fig.suptitle("Sersic n = 1, q = 0.4, PA = 30 (north up, east left): Fourier and bending modes")
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/m12_shape_gallery.png", dpi=90, bbox_inches="tight")
print("saved outputs/m12_shape_gallery.png")
