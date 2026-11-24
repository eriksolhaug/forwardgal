"""Milestone 7a: images of the quasar's source position, vs Lenstool and the data.

1. For each observed image (1A, 1B, 1C from the arcs file), back-trace it to
   beta and find all its images, as Lenstool does in image.dat. Compare our
   predicted images with Lenstool's predictions.
2. From the PSF-fitted JWST position of 1A (M6b), predict 1B and 1C and
   compare with their observed (arcs-file) positions.

Run from the repository root:
    python scripts/m7a_quasar_images.py
"""

import time
from pathlib import Path

import numpy as np

import forwardgal  # noqa: F401
from forwardgal.images import find_images
from forwardgal.lens import load_deflection
from forwardgal.quasar import image_fixed_quasar

MODEL = Path.home() / "Research/Tools/lenstool_tools/single_model/dec19a_model_solhaug"
field = load_deflection(
    "data/lensmodel/bestopt_1.524_dplx.fits", "data/lensmodel/bestopt_1.524_dply.fits", z_ref=1.524
)
OBS = {"1A": (178.3352456, 7.9335292), "1B": (178.3317916, 7.9364979), "1C": (178.3289710, 7.9369786)}

# Lenstool image.dat: for each observed image (label), the images it predicts.
lt_pred = {}
for line in (MODEL / "image.dat").read_text().splitlines():
    p = line.split()
    if p and p[0] in OBS:
        lt_pred.setdefault(p[0], []).append(field.sky_to_theta(float(p[1]), float(p[2])))


def nearest(theta, candidates):
    d = [np.hypot(theta[0] - float(c[0]), theta[1] - float(c[1])) for c in candidates]
    k = int(np.argmin(d))
    return k, d[k]


print("1. predicted images from each observed image, ours vs Lenstool image.dat")
for name, (ra, dec) in OBS.items():
    tx, ty = field.sky_to_theta(ra, dec)
    bx, by = field.backtrace(tx, ty)
    t0 = time.perf_counter()
    imgs = find_images(field, float(bx), float(by))
    dt = time.perf_counter() - t0
    print(f"\n  from {name}: beta = ({float(bx):+.4f}, {float(by):+.4f})\", {len(imgs)} images found ({dt:.2f} s)")
    for im in imgs:
        k, d_lt = nearest(im.theta, lt_pred[name])
        obs_name, d_obs = min(((n, np.hypot(im.theta[0] - float(field.sky_to_theta(*OBS[n])[0]),
                                            im.theta[1] - float(field.sky_to_theta(*OBS[n])[1]))) for n in OBS),
                              key=lambda t: t[1])
        print(f"    theta = ({im.theta[0]:+8.3f}, {im.theta[1]:+8.3f})  mu = {im.mu:+8.2f}   "
              f"to Lenstool prediction {d_lt:.3f}\"   to observed {obs_name} {d_obs:.3f}\"")

print("\n2. from the PSF-fitted JWST position of 1A (image-fixed quasar, M6b)")
q = image_fixed_quasar(178.3352306, 7.9335254, field)
print(q.describe())
for im in find_images(field, *q.beta):
    name, d = min(((n, np.hypot(im.theta[0] - float(field.sky_to_theta(*OBS[n])[0]),
                                im.theta[1] - float(field.sky_to_theta(*OBS[n])[1]))) for n in OBS),
                  key=lambda t: t[1])
    print(f"    theta = ({im.theta[0]:+8.3f}, {im.theta[1]:+8.3f})  mu = {im.mu:+8.2f}   nearest observed {name}: {d:.3f}\"")
