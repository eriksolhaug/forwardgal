"""Cosmology settings and the D_ls/D_s scaling of deflection angles.

For a single lens plane at z_lens, a deflection map computed for a source at
z_ref is converted to a source at z_src by

    alpha(z_src) = f * alpha(z_ref),   f = (D_ls/D_s)(z_src) / (D_ls/D_s)(z_ref),

with D_ls the angular-diameter distance from lens to source and D_s from
observer to source. H0 cancels in f; only Omega_M and Omega_Lambda matter.
"""

from __future__ import annotations

from dataclasses import dataclass

from astropy.cosmology import LambdaCDM


@dataclass(frozen=True)
class Cosmology:
    """Lambda-CDM cosmology (dark energy with w = -1).

    Attributes
    ----------
    H0 : float
        Hubble constant in km/s/Mpc. Only affects physical sizes (kpc), not
        the deflection scaling.
    Om0 : float
        Matter density Omega_M today.
    Ode0 : float
        Dark-energy density Omega_Lambda today (Lenstool: omegaX).
    """

    H0: float = 70.0
    Om0: float = 0.3
    Ode0: float = 0.7

    def to_astropy(self):
        return LambdaCDM(H0=self.H0, Om0=self.Om0, Ode0=self.Ode0)


def dls_over_ds(z_lens, z_src, cosmo):
    """Distance ratio D_ls / D_s (dimensionless) for a source behind the lens.

    Raises ValueError if z_src <= z_lens (the source is not lensed).
    """
    if z_src <= z_lens:
        raise ValueError(f"z_src = {z_src} is not behind z_lens = {z_lens}")
    c = cosmo.to_astropy()
    d_ls = c.angular_diameter_distance(z_lens, z_src)  # between two redshifts
    d_s = c.angular_diameter_distance(z_src)
    return float(d_ls / d_s)


def deflection_scale(z_lens, z_ref, z_src, cosmo):
    """Factor f that converts deflections computed for z_ref to z_src.

    f = (D_ls/D_s)(z_src) / (D_ls/D_s)(z_ref); f = 1 exactly if z_src == z_ref.
    """
    if z_src == z_ref:
        return 1.0
    return dls_over_ds(z_lens, z_src, cosmo) / dls_over_ds(z_lens, z_ref, cosmo)
