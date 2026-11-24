"""Cross-check against GALFIT model images (independent code; test only).

GALFIT is found via the environment variable FORWARDGAL_GALFIT (path to the
executable) or as ``galfit`` on the PATH; the tests are skipped otherwise.

GALFIT pixel frame -> forwardgal sky frame (north-up image, no rotation):
    x_east  = -(col - c) * pixscale,   y_north = (row - c) * pixscale,
with c the 0-based index of GALFIT's 1-based centre pixel. GALFIT's PA
(Up = 0, Left = 90) is then the same number as forwardgal's PA east of north.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest
from astropy.io import fits

import forwardgal  # noqa: F401
# [M12e paused] from forwardgal.profiles.sersic import Spiral, elliptical_radius, sersic, sersic_total_flux
from forwardgal.profiles.sersic import Bending, elliptical_radius, sersic  # [M12e stand-in]

GALFIT = os.environ.get("FORWARDGAL_GALFIT") or shutil.which("galfit")
needs_galfit = pytest.mark.skipif(not GALFIT, reason="GALFIT not found (set FORWARDGAL_GALFIT)")

N = 301          # image size (pixels)
CENTRE = 151     # GALFIT centre, 1-based -> index 150
PIXSCALE = 0.1   # arcsec / pixel
ZP = 20.0


def galfit_model(workdir, mag, re_pix, n, q, pa_deg, modes=None, spiral_pix=None):
    """GALFIT (P = 1, model only, no PSF) image of one Sersic. ``modes`` as in
    forwardgal: (m, a, phi_deg). ``spiral_pix``: (kind, r_in, r_out, theta_out,
    shape) with radii in PIXELS (GALFIT R0-R4; R9 = R10 = 0). Returns an
    (N, N) array indexed [row, col]."""
    blank = workdir / "blank.fits"
    fits.writeto(blank, np.zeros((N, N)), overwrite=True)
    fm = "" if modes is None else "\n".join(f"F{m}) {a} {p} 0 0" for m, a, p in zip(*modes))
    # [M12e paused] if spiral_pix is not None:
        # [M12e paused] kind, r_in, r_out, th, shape = spiral_pix
        # [M12e paused] fm += (f"\nR0) {kind}\nR1) {r_in} 0\nR2) {r_out} 0\nR3) {th} 0\nR4) {shape} 0"
               # [M12e paused] f"\nR9) 0 0\nR10) 0 0")
    feed = f"""A) {blank}
B) {workdir / 'model.fits'}
C) none
D) none
E) 1
F) none
G) none
H) 1 {N} 1 {N}
I) 100 100
J) {ZP}
K) {PIXSCALE} {PIXSCALE}
O) regular
P) 1
0) sersic
1) {CENTRE} {CENTRE} 0 0
3) {mag} 0
4) {re_pix} 0
5) {n} 0
9) {q} 0
10) {pa_deg} 0
{fm}
Z) 0
"""
    (workdir / "model.feedme").write_text(feed)
    subprocess.run([GALFIT, "model.feedme"], cwd=workdir, capture_output=True, check=True)
    with fits.open(workdir / "model.fits") as h:
        return np.array(h[0].data, float)


def forwardgal_points(re_pix, n, q, pa_deg, modes=None, spiral_pix=None):
    """forwardgal point values at the pixel centres, I_e = 1; also returns the
    elliptical radius in pixels (to exclude GALFIT's sub-sampled centre)."""
    rows, cols = np.indices((N, N))
    c = CENTRE - 1
    x, y = -(cols - c) * PIXSCALE, (rows - c) * PIXSCALE
    # [M12e paused] spiral = None
    # [M12e paused] if spiral_pix is not None:
        # [M12e paused] kind, r_in, r_out, th, shape = spiral_pix
        # [M12e paused] shape = shape * PIXSCALE if kind == "log" else shape  # r_ws is a radius; alpha is not
        # [M12e paused] spiral = Spiral(kind, r_in * PIXSCALE, r_out * PIXSCALE, th, shape)
    # [M12e paused] img = np.asarray(sersic(x, y, 0.0, 0.0, 1.0, re_pix * PIXSCALE, n, q, pa_deg, modes=modes, spiral=spiral))
    img = np.asarray(sersic(x, y, 0.0, 0.0, 1.0, re_pix * PIXSCALE, n, q, pa_deg, modes=modes))  # [M12e stand-in]
    R_pix = np.asarray(elliptical_radius(x, y, 0.0, 0.0, q, pa_deg)) / PIXSCALE
    return img, R_pix


CASES = [
    dict(re_pix=20.0, n=1.0, q=0.4, pa_deg=-60.0, modes=((1, 3, 4), (0.15, 0.05, 0.08), (70.0, -30.0, 25.0))),
    dict(re_pix=20.0, n=2.5, q=0.8, pa_deg=120.0, modes=((1, 4), (0.1, -0.06), (-110.0, 5.0))),
]


@needs_galfit
@pytest.mark.parametrize("case", CASES)
def test_fourier_modes_match_galfit(case, tmp_path):
    g = galfit_model(tmp_path, mag=15.0, **case)
    ours, R_pix = forwardgal_points(**case)
    ok = R_pix > 8  # GALFIT sub-samples the centre; compare point values outside
    k = (g[ok] * ours[ok]).sum() / (ours[ok] ** 2).sum()  # best-fitting I_e
    rel = np.abs(g[ok] / (k * ours[ok]) - 1)
    assert np.median(rel) < 1e-3, np.median(rel)
    assert np.abs(g[ok] - k * ours[ok]).max() / g[ok].max() < 5e-3


@needs_galfit
def test_galfit_magnitude_is_total_flux_of_perturbed_profile(tmp_path):
    """GALFIT's magnitude includes the Fourier modes: the image sums to
    10^(-0.4 (mag - ZP)) with or without them (n = 1: < 0.1% outside the box)."""
    case = dict(CASES[0])
    plain = galfit_model(tmp_path, 15.0, case["re_pix"], case["n"], case["q"], case["pa_deg"])
    modes = galfit_model(tmp_path, 15.0, **case)
    expected = 10 ** (-0.4 * (15.0 - ZP))
    assert plain.sum() == pytest.approx(expected, rel=2e-3)
    assert modes.sum() == pytest.approx(expected, rel=2e-3)


# [M12e paused] # --- spiral rotation (M12e) -------------------------------------------------

# [M12e paused] SPIRAL_CASES = [
    # [M12e paused] dict(re_pix=60.0, n=0.5, q=0.3, pa_deg=0.0, spiral_pix=("power", 20, 100, 90, 0.0)),
    # [M12e paused] dict(re_pix=60.0, n=0.5, q=0.4, pa_deg=-50.0, spiral_pix=("power", 40, 80, -120, 0.5)),
    # [M12e paused] dict(re_pix=40.0, n=1.0, q=0.35, pa_deg=-20.0, spiral_pix=("power", 25, 110, 300, 1.2)),
    # [M12e paused] dict(re_pix=60.0, n=0.5, q=0.3, pa_deg=10.0, spiral_pix=("log", 10, 120, -150, 5.0)),
    # [M12e paused] dict(re_pix=50.0, n=1.0, q=0.4, pa_deg=10.0, spiral_pix=("power", 20, 100, 90, 0.0),
         # [M12e paused] modes=((1, 3), (0.1, 0.05), (40.0, 10.0))),
# [M12e paused] ]


# [M12e paused] @needs_galfit
# [M12e paused] @pytest.mark.parametrize("case", SPIRAL_CASES)
# [M12e paused] def test_spiral_rotation_matches_galfit(case, tmp_path):
    # [M12e paused] g = galfit_model(tmp_path, mag=15.0, **case)
    # [M12e paused] ours, R_pix = forwardgal_points(**case)
    # [M12e paused] rows, cols = np.indices((N, N))
    # [M12e paused] # GALFIT sub-samples the centre: for n = 1 its inner pixels differ by ~1e-3 of the
    # [M12e paused] # peak out to ~10 px WITH OR WITHOUT a spiral (measured 2026-11-16), so compare beyond 15 px.
    # [M12e paused] ok = np.hypot(rows - (CENTRE - 1), cols - (CENTRE - 1)) > 15
    # [M12e paused] k = (g[ok] * ours[ok]).sum() / (ours[ok] ** 2).sum()
    # [M12e paused] assert np.abs(g[ok] - k * ours[ok]).max() / g[ok].max() < 1e-3
    # [M12e paused] assert np.median(np.abs(g[ok] / (k * ours[ok]) - 1)) < 2e-3


# [M12e paused] def test_spiral_rotation_conserves_total_flux():
    # [M12e paused] """(r, phi) -> (r, phi + theta(r)) moves light along circles only."""
    # [M12e paused] h = 0.02
    # [M12e paused] x, y = np.meshgrid(np.arange(-8, 8, h) + h / 2, np.arange(-8, 8, h) + h / 2)
    # [M12e paused] sp = Spiral("power", 0.3, 1.5, 160.0, 0.5)
    # [M12e paused] img = np.asarray(sersic(x, y, 0.0, 0.0, 1.0, 0.8, 1.0, 0.3, 20.0, spiral=sp))
    # [M12e paused] assert img.sum() * h**2 == pytest.approx(float(sersic_total_flux(1.0, 0.8, 1.0, 0.3)), rel=1e-4)


# --- bending modes (M12f) -----------------------------------------------------


def galfit_bent(workdir, re_pix, n, q, pa_deg, bends_pix, modes=None):
    """GALFIT model with bending modes ``bends_pix`` = {m: amplitude in pixels}."""
    blank = workdir / "blank.fits"
    fits.writeto(blank, np.zeros((N, N)), overwrite=True)
    extra = "\n".join(f"B{m}) {a} 0" for m, a in bends_pix.items())
    if modes is not None:
        extra += "\n" + "\n".join(f"F{m}) {a} {p} 0 0" for m, a, p in zip(*modes))
    (workdir / "b.feedme").write_text(f"""A) {blank}
B) {workdir / 'b.fits'}
C) none
D) none
E) 1
F) none
G) none
H) 1 {N} 1 {N}
I) 100 100
J) {ZP}
K) {PIXSCALE} {PIXSCALE}
O) regular
P) 1
0) sersic
1) {CENTRE} {CENTRE} 0 0
3) 15.0 0
4) {re_pix} 0
5) {n} 0
9) {q} 0
10) {pa_deg} 0
{extra}
Z) 0
""")
    subprocess.run([GALFIT, "b.feedme"], cwd=workdir, capture_output=True, check=True)
    with fits.open(workdir / "b.fits") as h:
        return np.array(h[0].data, float)


BEND_CASES = [  # (re_pix, n, q, pa_deg, {m: a_pix}, modes); PA inside and outside GALFIT's (-90, 90]
    (40.0, 1.0, 0.4, 30.0, {2: 3.0}, None),
    (50.0, 0.5, 0.3, -60.0, {1: 2.0, 2: -4.0, 3: 1.5}, None),
    (30.0, 1.0, 0.5, 120.0, {1: 2.0, 2: -3.0, 3: 1.0}, None),
    (30.0, 1.0, 0.5, -150.0, {2: 3.0, 4: 2.0, 5: 1.0}, None),
    (40.0, 1.0, 0.4, 10.0, {2: 3.0}, ((1, 3), (0.1, 0.05), (40.0, 10.0))),
]


@needs_galfit
@pytest.mark.parametrize("case", BEND_CASES)
def test_bending_modes_match_galfit(case, tmp_path):
    """GALFIT's convention is pa_ref = 0; amplitudes in pixels -> arcsec."""
    re_pix, n, q, pa, bends, modes = case
    g = galfit_bent(tmp_path, re_pix, n, q, pa, bends, modes)
    rows, cols = np.indices((N, N))
    c = CENTRE - 1
    x, y = -(cols - c) * PIXSCALE, (rows - c) * PIXSCALE
    bending = Bending(tuple(bends), [a * PIXSCALE for a in bends.values()], 0.0)
    ours = np.asarray(sersic(x, y, 0.0, 0.0, 1.0, re_pix * PIXSCALE, n, q, pa, modes=modes, bending=bending))
    ok = np.hypot(rows - c, cols - c) > 15  # beyond GALFIT's sub-sampled centre (see the spiral note)
    k = (g[ok] * ours[ok]).sum() / (ours[ok] ** 2).sum()
    assert np.abs(g[ok] - k * ours[ok]).max() / g[ok].max() < 5e-5  # measured 4e-7 (n = 1) to 1.6e-5 (n = 0.5)


def test_bending_conserves_flux_and_moves_the_seam():
    """Shear-like map: total flux unchanged. Any pa_ref within 90 deg of the
    PA gives the same frame; pa_ref + 180 turns the frame by 180 deg, which
    flips the even modes and leaves the odd ones."""
    h = 0.02
    x, y = np.meshgrid(np.arange(-8, 8, h) + h / 2, np.arange(-8, 8, h) + h / 2)
    geo = (0.0, 0.0, 1.0, 0.6, 1.0, 0.4, 25.0)
    b = Bending((2, 3), [0.05, -0.01], 25.0)  # moderate: the bent light stays inside the box
    img = np.asarray(sersic(x, y, *geo, bending=b))
    from forwardgal.profiles.sersic import sersic_total_flux
    assert img.sum() * h**2 == pytest.approx(float(sersic_total_flux(1.0, 0.6, 1.0, 0.4)), rel=1e-4)
    same = np.asarray(sersic(x, y, *geo, bending=Bending((2, 3), [0.05, -0.01], 25.0 + 80.0)))
    np.testing.assert_allclose(same, img, rtol=1e-12)
    turned = np.asarray(sersic(x, y, *geo, bending=Bending((2, 3), [-0.05, -0.01], 25.0 + 180.0)))
    np.testing.assert_allclose(turned, img, rtol=1e-10, atol=1e-14)
