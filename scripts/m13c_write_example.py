"""Milestone 13c: write the 1A bulge + disk set-up (M10d) as a model YAML file.

The example mask (scripts/common.py: an ellipse in cutout pixels) is written
as a polygon in RA, Dec. All paths are written absolute, so the file works
from any folder. Writes examples/1A_bulge_disk.yaml, then loads it back and
prints the set-up and chi^2 at the starting values.

Run from the repository root:
    python scripts/m13c_write_example.py
"""

from pathlib import Path

import numpy as np

import forwardgal  # noqa: F401
from common import (CUTOUT_ARCSEC, DPLX, DPLY, IMAGE, MASK_A, MASK_B, MASK_CENTRE_RC, MASK_PA, PSF_FILE, QSO_1A,
                    load_example_cutout)
from forwardgal.components import ModelParams, sersic_component, sky_component
from forwardgal.config import pixels_to_radec
from forwardgal.regions import ellipse_polygon
from forwardgal.serialize import model_to_dict

ROOT = Path(__file__).resolve().parents[1]


def example_config():
    img = load_example_cutout()
    poly_rc = ellipse_polygon(MASK_CENTRE_RC, MASK_A, MASK_B, MASK_PA)
    bx, by = -5.7323, 0.1945  # quasar 1A in the source plane (M6b), rounded
    params = ModelParams([
        sersic_component("bulge", "source", bx + 0.005, by - 0.005, R_e=0.05, n=4.0, q=0.8, pa_deg=30.0),
        sersic_component("disk", "source", bx + 0.01, by - 0.01, R_e=0.25, n=1.0, q=0.5, pa_deg=30.0),
        sky_component(),
    ])
    return {
        "label": "1A-bulge-disk",
        "data": {"image": {"file": str(IMAGE), "ext": "SCI"}, "sigma": {"file": str(IMAGE), "ext": "ERR"},
                 "centre": [float(x) for x in QSO_1A], "size_arcsec": CUTOUT_ARCSEC, "zeropoint": None},
        "lens": {"deflection_x": str(ROOT / DPLX), "deflection_y": str(ROOT / DPLY), "z_ref": 1.524,
                 "x_points_west": True, "interpolation": "bicubic"},
        "cosmology": {"H0": 70.0, "Om0": 0.3, "Ode0": 0.7, "z_lens": 0.4301, "z_src": 1.524},
        "mask": {"polygon_radec": [[float(a), float(b)] for a, b in pixels_to_radec(img.wcs, poly_rc)],
                 "margin_px": 35},
        "psf": {"file": str(ROOT / PSF_FILE), "ext": 0, "reference": "centroid"},
        "quasar": {"name": "QSO 1A", "mode": "image_fixed", "ra": float(QSO_1A[0]), "dec": float(QSO_1A[1]),
                   "max_snap": 0.1},
        "render": {"subsample": 5, "patch": 7, "exact_pixels": 200, "gl_nodes": 12, "bend_split": 4},
        "fit": {"method": "lm", "n_starts": 5, "seed": 0},
        "model": model_to_dict(params),
    }


if __name__ == "__main__":
    from forwardgal.config import load
    from forwardgal.serialize import write_yaml

    out = ROOT / "examples" / "1A_bulge_disk.yaml"
    out.parent.mkdir(exist_ok=True)
    write_yaml(example_config(), out, header="forwardgal model file (M13c example): quasar 1A host, F115W,\n"
                                              "bulge + disk + quasar + sky. Written by scripts/m13c_write_example.py.")
    print(f"wrote {out}")
    s = load(out)
    c2, dof = s.model.chi2_dof(s.model.params.start())
    print(s.region.describe())
    print(f"mask pixels {int(s.region.mask.sum())}; chi2/dof at the start = {c2:.1f}/{dof}")
    print(np.round(s.model.solve(s.model.params.start())[1], 4))
