"""Milestone 2c: back-trace the quasar images to the source plane.

If the deflection maps and the coordinate conventions are right, the three
images 1A, 1B, 1C must map to (nearly) one source position beta. Their scatter
should match Lenstool's own source-plane rms for this system (chires.dat,
system 1: rmss = 0.086").

Run from the repository root:
    python scripts/m2c_backtrace.py
"""

import numpy as np

from forwardgal.lens import load_deflection

field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits",
    "data/lensmodel/bestopt_1.524_dply.fits",
    z_ref=1.524,
)

QSO = {  # arcs/1153arc_v2_zvary.dat, system 1 (z = 1.524)
    "1A": (178.3352456, 7.9335292),
    "1B": (178.3317916, 7.9364979),
    "1C": (178.3289710, 7.9369786),
}

names = list(QSO)
ra, dec = np.array([QSO[n] for n in names]).T
theta_x, theta_y = field.sky_to_theta(ra, dec)
alpha_x, alpha_y = field.alpha(theta_x, theta_y)
beta_x, beta_y = field.backtrace(theta_x, theta_y)

print("image   theta_x   theta_y   alpha_x   alpha_y    beta_x    beta_y   [arcsec]")
for k, n in enumerate(names):
    print(
        f"{n:5s} {theta_x[k]:+9.3f} {theta_y[k]:+9.3f} {alpha_x[k]:+9.3f} "
        f"{alpha_y[k]:+9.3f} {beta_x[k]:+9.3f} {beta_y[k]:+9.3f}"
    )

# Mean source position and rms distance of the individual betas from it.
bx_mean, by_mean = beta_x.mean(), beta_y.mean()
dist = np.hypot(beta_x - bx_mean, beta_y - by_mean)
rms = np.sqrt(np.mean(dist**2))
print(f"\nmean beta          : ({bx_mean:+.3f}, {by_mean:+.3f}) arcsec")
print("distance from mean : " + ", ".join(f"{n} {d:.3f}" for n, d in zip(names, dist)) + " arcsec")
print(f"rms scatter        : {rms:.3f} arcsec   (Lenstool rmss for system 1: 0.086)")
