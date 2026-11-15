"""Milestone 2d: compute the D_ls/D_s rescaling of the deflection maps.

The settings below are typed in by the user (later: the GUI Settings tab).
They should match the cosmology the lens model was built with.

Run from the repository root:
    python scripts/m2d_cosmology.py
"""

from forwardgal.cosmology import Cosmology, deflection_scale, dls_over_ds

# --- user settings ---------------------------------------------------------
COSMO = Cosmology(H0=70.0, Om0=0.3, Ode0=0.7)
Z_LENS = 0.4301
Z_REF = 1.524  # source redshift the deflection maps were computed for
Z_SRC = 1.524  # quasar redshift
# ---------------------------------------------------------------------------

print(f"cosmology         : H0 = {COSMO.H0} km/s/Mpc, Omega_M = {COSMO.Om0}, Omega_Lambda = {COSMO.Ode0}")
print(f"z_lens, z_ref     : {Z_LENS}, {Z_REF}")
print(f"D_ls/D_s at z_ref : {dls_over_ds(Z_LENS, Z_REF, COSMO):.5f}")
print(f"z_src = {Z_SRC} -> alpha scale factor f = {deflection_scale(Z_LENS, Z_REF, Z_SRC, COSMO):.5f}")
print()
for z in [1.0, 1.939, 3.0]:
    print(f"(for comparison) z_src = {z:5.3f} -> f = {deflection_scale(Z_LENS, Z_REF, z, COSMO):.5f}")
