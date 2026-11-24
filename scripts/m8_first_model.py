"""Milestone 8: first complete forward model of the 1A cutout (amplitudes only).

Components (templates at unit brightness):
  cluster: the bright SW cluster galaxy as an IMAGE-PLANE (unlensed) Sersic
           (shape guessed, centre at its brightest pixel), rendered on the
           whole render box (its core lies outside the render pixels);
  host   : lensed Sersic (shape GUESSED, not fitted: that is M10), I_e = 1,
           rendered on the render pixels and PSF-convolved;
  quasar : image-fixed 1A at its PSF-fitted position (M6b), unit-flux PSF
           placed with zero fractional shift (the PSF was made for it);
  sky    : constant.
Amplitudes are solved inside an example mask (the GUI will provide the real
one) with NNLS (fluxes >= 0, sky free) and, for error bars, unconstrained.

Run from the repository root:
    python scripts/m8_first_model.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

import forwardgal  # noqa: F401
from common import load_example_cutout, load_field, example_mask, PSF_FILE, QSO_1A
from forwardgal.likelihood import amplitude_covariance, design_matrix, profile_chi2
from forwardgal.profiles.sersic import unit_flux_ie
from forwardgal.psf import load_psf, make_convolver, make_point_source_renderer
from forwardgal.quasar import image_fixed_quasar, snap_to_psf
from forwardgal.regions import fit_region
from forwardgal.render import render_lensed_sersic
from forwardgal.traced import trace_grid

# 9" cutout centred on the PSF-fitted quasar 1A (M6b).
img = load_example_cutout()
field = load_field()
psf = load_psf(PSF_FILE)
yy, xx = np.indices(psf.kernel.shape)
PSF_REF = ((psf.kernel * yy).sum(), (psf.kernel * xx).sum())

# Example mask (the GUI will provide the real one): ellipse aligned with the
# arc, PA 166 deg east of north (flux-weighted moments of the arc light),
# centred on the arc light, semi-axes 100 x 32 px (~10,000 px). It stays
# 0.88" clear of the nearest critical curve (around the SW cluster galaxy).
ny, nx = img.data.shape
region = fit_region(example_mask((ny, nx)), margin=35)
print(region.describe())
r0, r1, c0, c1 = region.box
shape = region.box_shape
grid = trace_grid(img.wcs, shape, field, subsample=5, origin=(r0, c0))
convolve = make_convolver(psf.kernel, shape)
pix = region.render_pixels_in_box()
mask = region.in_box(region.mask)
data, sigma = region.in_box(img.data), region.in_box(img.sigma)

qso = image_fixed_quasar(*QSO_1A, field)
print("\n" + qso.describe())
# The PSF was made for 1A: snap its measured position (noise ~0.03 px) to the
# whole-pixel-aligned stamp, so the PSF goes in with no fractional shift.
QSO_RC, snap = snap_to_psf(qso.grid_position(img.wcs, origin=(r0, c0)), PSF_REF, psf.centre)
print(f"quasar snapped to the PSF's own position: moved {snap:.3f} px ({snap * img.pixscale * 1000:.1f} mas)")
host = dict(x0=qso.beta[0] + 0.01, y0=qso.beta[1] - 0.01, R_e=0.12, n=1.0, q=0.6, pa_deg=30.0)  # GUESS
print(f"host shape (guessed): {host}")

# The SW cluster galaxy: unlensed, so its parameters are in the image plane.
from scipy.ndimage import gaussian_filter  # noqa: E402

from forwardgal.traced import pixel_theta  # noqa: E402

sm = gaussian_filter(img.data, 2.0)
sw = np.zeros_like(sm, bool)
sw[: ny // 2, nx // 2 :] = True  # south-west quadrant (south = low rows, west = high cols)
cr, cc = np.unravel_index(np.argmax(np.where(sw, sm, -np.inf)), sm.shape)
cl_x, cl_y = (float(v) for v in pixel_theta(img.wcs, field, np.array(cr), np.array(cc)))
cluster = dict(x0=cl_x, y0=cl_y, R_e=0.5, n=4.0, q=0.8, pa_deg=0.0)  # GUESS
print(f"cluster galaxy (image plane): brightest pixel [{cr}, {cc}] -> theta = ({cl_x:+.3f}, {cl_y:+.3f})\"; shape guessed")

templates = [
    # Sersic templates normalized to unit total INTRINSIC flux (M9d): their
    # amplitudes are total fluxes in pixel-sum units, m = -2.5 log10(a) + ZP.
    convolve(render_lensed_sersic(grid, host["x0"], host["y0"],
                                  float(unit_flux_ie(host["R_e"], host["n"], host["q"], img.pixscale)),
                                  host["R_e"], host["n"],
                                  host["q"], host["pa_deg"], pixels=pix)),
    make_point_source_renderer(psf.kernel, shape, PSF_REF)(*QSO_RC),
    convolve(render_lensed_sersic(grid, cluster["x0"], cluster["y0"],
                                  float(unit_flux_ie(cluster["R_e"], cluster["n"], cluster["q"], img.pixscale)),
                                  cluster["R_e"], cluster["n"],
                                  cluster["q"], cluster["pa_deg"], plane="image")),
    jnp.ones(shape),
]
names = ["host flux, intrinsic [pixel sum]  (source plane)", "quasar 1A flux [pixel sum]",
         "cluster gal. flux [pixel sum]  (image plane)", "sky [MJy/sr]"]
T = design_matrix(templates, mask)
d, s = jnp.asarray(data[mask]), jnp.asarray(sigma[mask])

c2_nn, a_nn = profile_chi2(T, d, s, method="nnls", nonneg=[True, True, True, False])
c2_ls, a_ls = profile_chi2(T, d, s, method="lstsq")
err = np.sqrt(np.diag(amplitude_covariance(T, s)))
dof = int(mask.sum()) - len(templates)
print(f"\n{'amplitude':38s} {'NNLS':>12s} {'unconstrained':>14s} {'+-':>10s}")
for k, n in enumerate(names):
    print(f"{n:38s} {float(a_nn[k]):12.5g} {float(a_ls[k]):14.5g} {err[k]:10.2g}")
print(f"chi2/dof (NNLS) = {float(c2_nn):.0f}/{dof} = {float(c2_nn) / dof:.2f}")

ZP = img.zeropoint
m_host_src = -2.5 * np.log10(float(a_nn[0])) + ZP  # amplitude = intrinsic total flux
m_host_obs = -2.5 * np.log10(float(a_nn[0] * templates[0][mask].sum())) + ZP
m_qso_obs = -2.5 * np.log10(float(a_nn[1])) + ZP
mu_1a = 1.0 / grid.area_ratio().ravel()[np.argmax(np.asarray(templates[1]))]
print(f"\nhost  : intrinsic (source-plane) m_tot = {m_host_src:.2f} AB; lensed light inside the mask = {m_host_obs:.2f} AB")
print(f"quasar: image 1A m = {m_qso_obs:.2f} AB; with mu(1A) = {mu_1a:.2f}: intrinsic m = {m_qso_obs + 2.5 * np.log10(abs(mu_1a)):.2f} AB")

model = sum(float(a_nn[k]) * np.asarray(templates[k]) for k in range(len(templates)))
resid = np.where(mask, (data - model) / sigma, np.nan)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
for ax, arr, title, kw in [
    (axes[0], data, "data", dict(norm=norm, cmap="gray")),
    (axes[1], model, "model (host + quasar + cluster galaxy + sky)", dict(norm=norm, cmap="gray")),
    (axes[2], resid, f"(data - model) / sigma in mask; chi2/dof = {float(c2_nn) / dof:.1f}", dict(cmap="RdBu_r", vmin=-10, vmax=10)),
]:
    im = ax.imshow(arr, origin="lower", **kw)
    ax.contour(mask, levels=[0.5], colors="C1", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
out = Path("outputs/m8_first_model.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nsaved figure: {out}")

# --- the source-plane model ---------------------------------------------------
from forwardgal.critical import caustics, critical_curves  # noqa: E402
from forwardgal.render import source_plane_image  # noqa: E402

host_src = dict(host, I_e=float(a_nn[0]) * float(unit_flux_ie(host["R_e"], host["n"], host["q"], img.pixscale)))
src_img, extent = source_plane_image([host_src], centre=qso.beta, size=2.0, pixscale=0.005)
beta_mask_x = grid.beta_sub_x.mean(axis=-1)[mask]  # traced centres of the mask pixels
beta_mask_y = grid.beta_sub_y.mean(axis=-1)[mask]
fig2, ax = plt.subplots(figsize=(6.5, 6))
im = ax.imshow(np.log10(np.maximum(src_img, 1e-3 * src_img.max())), origin="lower", extent=extent, cmap="magma")
# Outline of the traced mask: source-plane pixels that receive at least one mask pixel.
edges_x = np.linspace(extent[1], extent[0], src_img.shape[1] + 1)
edges_y = np.linspace(extent[2], extent[3], src_img.shape[0] + 1)
hit, _, _ = np.histogram2d(beta_mask_y, beta_mask_x, bins=[edges_y, edges_x])
from scipy.ndimage import binary_closing  # noqa: E402

covered = binary_closing(hit[:, ::-1] > 0, iterations=3)  # columns run west, as in src_img
ax.contour(covered, levels=[0.5], colors="c", linewidths=1.5, extent=extent, origin="lower")
ax.plot([], [], "c-", label="mask, traced to the source plane")
for k, c in enumerate(caustics(field, critical_curves(field))):
    ax.plot(c[:, 0], c[:, 1], "w-", lw=0.8, label="caustics" if k == 0 else None)
ax.plot(*qso.beta, "*", color="lime", ms=14, label="quasar (point source)")
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("beta_x [arcsec, +east]")
ax.set_ylabel("beta_y [arcsec, +north]")
ax.set_title("source-plane model: log10 I (host shape guessed)", fontsize=9)
ax.legend(loc="lower left", fontsize=8)
fig2.colorbar(im, ax=ax, fraction=0.046, label="log10 I [MJy/sr]")
out2 = Path("outputs/m8_source_plane.png")
fig2.savefig(out2, dpi=110, bbox_inches="tight")
area_src = np.ptp(beta_mask_x) * np.ptp(beta_mask_y)
print(f"saved figure: {out2}")
print(f"mask traced to the source plane spans {np.ptp(beta_mask_x):.2f}\" x {np.ptp(beta_mask_y):.2f}\" (bounding box)")
plt.show()
