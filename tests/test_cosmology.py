import numpy as np
import pytest

from forwardgal.cosmology import Cosmology, deflection_scale, dls_over_ds

EDS = Cosmology(H0=70.0, Om0=1.0, Ode0=0.0)  # Einstein-de Sitter


def dls_ds_eds(z_l, z_s):
    """Closed form for Einstein-de Sitter: 1 - chi(z_l)/chi(z_s)."""
    chi = lambda z: 1.0 - (1.0 + z) ** -0.5  # noqa: E731  (in units of 2c/H0)
    return 1.0 - chi(z_l) / chi(z_s)


@pytest.mark.parametrize("z_l, z_s", [(0.4301, 1.524), (0.4301, 3.0), (0.2, 0.5)])
def test_dls_ds_matches_eds_closed_form(z_l, z_s):
    assert dls_over_ds(z_l, z_s, EDS) == pytest.approx(dls_ds_eds(z_l, z_s), rel=1e-6)


def test_h0_cancels():
    a = deflection_scale(0.4301, 1.524, 3.0, Cosmology(H0=50.0))
    b = deflection_scale(0.4301, 1.524, 3.0, Cosmology(H0=90.0))
    assert a == pytest.approx(b, rel=1e-12)


def test_scale_is_one_at_reference_redshift():
    assert deflection_scale(0.4301, 1.524, 1.524, Cosmology()) == 1.0


def test_scale_grows_with_source_redshift():
    c = Cosmology()
    assert deflection_scale(0.4301, 1.524, 1.0, c) < 1.0
    assert deflection_scale(0.4301, 1.524, 3.0, c) > 1.0


def test_source_in_front_of_lens_rejected():
    with pytest.raises(ValueError):
        dls_over_ds(0.4301, 0.3, Cosmology())


def test_rescaled_field_multiplies_maps():
    from astropy.wcs import WCS

    from forwardgal.lens import DeflectionField

    ones = np.ones((3, 3))
    f = DeflectionField(ones, 2 * ones, WCS(naxis=2), -0.1, 0.1, 1.0, 1.0, z_ref=1.524)
    c = Cosmology()
    g = f.rescaled(0.4301, 3.0, c)
    k = deflection_scale(0.4301, 1.524, 3.0, c)
    np.testing.assert_allclose(g.alpha_x, k * ones)
    np.testing.assert_allclose(g.alpha_y, 2 * k * ones)
    assert g.z_ref == 3.0 and f.alpha_x[0, 0] == 1.0  # original untouched
