"""Shared example setup for the milestone scripts (COOL J1153, F115W, image 1A).

One place for the data paths, the example cutout and the example mask, so
that all scripts use the same ones. Scripts are run from the repository root
(``python scripts/<name>.py``); Python then finds this module next to them.
"""

from pathlib import Path

from forwardgal.data import load_cutout
from forwardgal.lens import load_deflection
from forwardgal.regions import ellipse_polygon, polygon_mask

IMAGE = Path.home() / "Research/Data_and_Analysis/DATA/CJ1153/jw06675-o001_t002_nircam_clear-f115w_i2d.fits"
LENSTOOL_MODEL = Path.home() / "Research/Tools/lenstool_tools/single_model/dec19a_model_solhaug"
DPLX = "data/lensmodel/bestopt_1.524_dplx.fits"
DPLY = "data/lensmodel/bestopt_1.524_dply.fits"
PSF_FILE = "data/rotated_jw06675-o001_t002_nircam_clear-f115w-QSOblueA_PSF_i2dtrim.fits"

QSO_1A = (178.3352306, 7.9335254)  # PSF-fitted position of quasar image 1A (M6b)
CUTOUT_ARCSEC = 9.0  # 300 x 300 px at 0.03"/px, centred on QSO_1A

# Example mask (the GUI will provide the real one): ellipse aligned with the arc,
# PA 166 deg east of north (flux-weighted moments of the arc light), centred on
# the arc light, semi-axes 100 x 32 px (~10,050 px); 0.88" clear of the nearest
# critical curve (around the SW cluster galaxy).
MASK_CENTRE_RC, MASK_A, MASK_B, MASK_PA = (144.2, 151.9), 100.0, 32.0, 166.0


def load_example_cutout():
    return load_cutout(IMAGE, "SCI", IMAGE, "ERR", *QSO_1A, size_arcsec=CUTOUT_ARCSEC)


def load_field(interpolation="bicubic"):
    return load_deflection(DPLX, DPLY, z_ref=1.524, interpolation=interpolation)


def example_mask(shape):
    return polygon_mask(shape, ellipse_polygon(MASK_CENTRE_RC, MASK_A, MASK_B, MASK_PA))
