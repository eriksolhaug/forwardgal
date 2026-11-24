"""Milestone 12a: forwardgal Fourier modes vs GALFIT 3 model images.

Renders one Sersic with modes m = 1, 3, 4 in GALFIT (model only) and in
forwardgal, prints the agreement and saves GALFIT | forwardgal | difference.
Needs GALFIT: set FORWARDGAL_GALFIT=/path/to/galfit.

Run from the repository root:
    FORWARDGAL_GALFIT=~/Research/Tools/galfit/galfit python scripts/m12a_galfit_fourier.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tests")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from test_galfit import CASES, GALFIT, forwardgal_points, galfit_model  # noqa: E402

if not GALFIT:
    sys.exit("GALFIT not found: set FORWARDGAL_GALFIT=/path/to/galfit")
case = CASES[0]
print("case:", case)
with tempfile.TemporaryDirectory() as d:
    g = galfit_model(Path(d), mag=15.0, **case)
ours, R_pix = forwardgal_points(**case)
ok = R_pix > 8  # GALFIT sub-samples the centre
k = (g[ok] * ours[ok]).sum() / (ours[ok] ** 2).sum()
rel = np.abs(g[ok] / (k * ours[ok]) - 1)
print(f"outside R = 8 px: median |GALFIT/ours - 1| = {np.median(rel):.1e}, "
      f"max |diff| / peak = {np.abs(g[ok] - k * ours[ok]).max() / g[ok].max():.1e}")
print(f"GALFIT image sum = {g.sum():.3f} (mag 15, ZP 20 -> 100 expected)")

s = slice(70, 231)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
lev = np.log10(g[s, s].max()) - np.arange(0, 4, 0.5)[::-1]
for a, img, t in [(ax[0], g, "GALFIT 3.0.5"), (ax[1], k * ours, "forwardgal")]:
    a.imshow(np.log10(np.maximum(img[s, s], 1e-8)), origin="lower", cmap="magma", vmin=lev[0] - 0.5)
    a.contour(np.log10(np.maximum(img[s, s], 1e-8)), levels=lev, colors="w", linewidths=0.6, linestyles="solid")
    a.set_title(t)
diff = np.where(ok, g - k * ours, np.nan)[s, s] / g.max()
im = ax[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-3e-3, vmax=3e-3)
ax[2].set_title("(GALFIT - forwardgal) / peak, R > 8 px")
fig.colorbar(im, ax=ax[2], fraction=0.046)
for a in ax:
    a.set_xlabel("column (north up, east left)")
fig.suptitle("Sersic + Fourier modes m = 1, 3, 4: " + ", ".join(
    f"a{m}={a:+.2f}, phi{m}={p:+.0f}" for m, a, p in zip(*case["modes"])))
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/m12a_galfit_fourier.png", dpi=100, bbox_inches="tight")
print("saved outputs/m12a_galfit_fourier.png")
