"""Milestone 5: critical curves and caustics from the deflection maps vs Lenstool.

Lenstool's ci.dat / ce.dat hold marching-squares segments: per line
"id  x_crit  y_crit  x_caustic  y_caustic" in arcsec relative to the
reference position. Both possible signs of Lenstool's x axis are tried; the
right one makes the curves coincide.

Run from the repository root:
    python scripts/m5_critical_curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import forwardgal  # noqa: F401
from forwardgal.critical import caustics, critical_curves
from forwardgal.lens import load_deflection

MODEL = Path.home() / "Research/Tools/lenstool_tools/single_model/dec19a_model_solhaug"
field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits", "data/lensmodel/bestopt_1.524_dply.fits", z_ref=1.524
)
crit = critical_curves(field)
caus = caustics(field, crit)
ours_crit, ours_caus = np.vstack(crit), np.vstack(caus)

lt = np.vstack([np.loadtxt(MODEL / f, comments="#")[:, 1:5] for f in ("ci.dat", "ce.dat")])
print(f"our critical-curve points: {len(ours_crit)} in {len(crit)} curves; Lenstool points: {len(lt)}")

QSO = {"1A": (178.3352456, 7.9335292), "1B": (178.3317916, 7.9364979), "1C": (178.3289710, 7.9369786)}
BETA_QSO = (-5.748, 0.124)

best = None
for sign, name in [(+1, "Lenstool x = +east"), (-1, "Lenstool x = west (x_east = -x)")]:
    lt_crit = np.column_stack([sign * lt[:, 0], lt[:, 1]])
    lt_caus = np.column_stack([sign * lt[:, 2], lt[:, 3]])
    # Distance from each of OUR points to the nearest Lenstool point, restricted
    # to where Lenstool computed curves (its field may be smaller than ours).
    d_crit, _ = cKDTree(lt_crit).query(ours_crit)
    d_caus, _ = cKDTree(lt_caus).query(ours_caus)
    inside = (np.abs(ours_crit[:, 0]) < np.abs(lt_crit[:, 0]).max()) & (np.abs(ours_crit[:, 1]) < np.abs(lt_crit[:, 1]).max())
    print(f"\n{name}:")
    print(f"  critical curves: median {np.median(d_crit[inside]):.3f}\", 95% {np.percentile(d_crit[inside], 95):.3f}\"")
    print(f"  caustics       : median {np.median(d_caus[inside]):.3f}\", 95% {np.percentile(d_caus[inside], 95):.3f}\"")
    if best is None or np.median(d_crit[inside]) < best[0]:
        best = (np.median(d_crit[inside]), sign, lt_crit, lt_caus)
_, sign, lt_crit, lt_caus = best
print(f"\nbest match: Lenstool x sign = {sign:+d} (x_east = {sign:+d} * x_Lenstool)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
ax1.plot(lt_crit[:, 0], lt_crit[:, 1], ".", color="0.6", ms=1.5, label="Lenstool ci/ce")
for k, c in enumerate(crit):
    ax1.plot(c[:, 0], c[:, 1], "C3-", lw=0.8, label="forwardgal" if k == 0 else None)
for name, (ra, dec) in QSO.items():
    tx, ty = field.sky_to_theta(ra, dec)
    ax1.plot(tx, ty, "k*", ms=9)
    ax1.annotate(name, (tx, ty), xytext=(4, 4), textcoords="offset points")
ax1.set_title("critical curves (image plane, z = 1.524)")
ax1.set_xlabel("theta_x [arcsec, +east]")
ax1.set_ylabel("theta_y [arcsec, +north]")
ax2.plot(lt_caus[:, 0], lt_caus[:, 1], ".", color="0.6", ms=1.5, label="Lenstool ci/ce")
for k, c in enumerate(caus):
    ax2.plot(c[:, 0], c[:, 1], "C0-", lw=0.8, label="forwardgal" if k == 0 else None)
ax2.plot(*BETA_QSO, "k*", ms=10, label="quasar (back-traced mean)")
ax2.set_title("caustics (source plane)")
ax2.set_xlabel("beta_x [arcsec, +east]")
ax2.set_ylabel("beta_y [arcsec, +north]")
for ax, lim in [(ax1, 35), (ax2, 25)]:
    ax.set_xlim(lim, -lim)  # east to the left
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
out = Path("outputs/m5_critical_curves.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved figure: {out}")
plt.show()
