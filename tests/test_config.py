"""M13c: a whole fit set-up from a configuration (YAML) file."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS

import forwardgal  # noqa: F401
from forwardgal.config import build, load, pixels_to_radec, radec_to_pixels
from forwardgal.serialize import write_yaml

ROOT = Path(__file__).resolve().parents[1]


def tan_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [178.3352306, 7.9335254]
    w.wcs.crpix = [151.0, 151.0]
    w.wcs.cd = [[-0.03 / 3600, 0.0], [0.0, 0.03 / 3600]]
    return w


def test_mask_polygon_radec_pixel_round_trip():
    rc = np.array([[10.0, 20.0], [150.0, 160.5], [290.2, 30.7]])
    back = radec_to_pixels(tan_wcs(), pixels_to_radec(tan_wcs(), rc))
    np.testing.assert_allclose(back, rc, atol=1e-8)


def test_radec_to_pixels_axes_and_zero_based():
    """crpix (1-based FITS) 151 -> 0-based index 150; east is left (decreasing col)."""
    w = tan_wcs()
    rc = radec_to_pixels(w, [[178.3352306, 7.9335254], [178.3352306 + 1.0 / 3600, 7.9335254]])
    np.testing.assert_allclose(rc[0], [150.0, 150.0], atol=1e-8)
    assert rc[1, 1] < rc[0, 1]  # 1" east -> smaller column
    assert rc[1, 0] == pytest.approx(150.0, abs=1e-3)


def test_unknown_section_rejected():
    with pytest.raises(ValueError, match="moddel"):
        build({"moddel": {}})


# --- real data: the M10d bulge + disk set-up, from Python and from YAML ------------

sys_path_scripts = str(ROOT / "scripts")


def _common():
    import sys

    if sys_path_scripts not in sys.path:
        sys.path.insert(0, sys_path_scripts)
    import common

    return common


def _have_data():
    try:
        c = _common()
    except Exception:
        return False
    return all(Path(p).exists() for p in (c.IMAGE, ROOT / c.DPLX, ROOT / c.DPLY, ROOT / c.PSF_FILE))


needs_data = pytest.mark.skipif(not _have_data(), reason="COOL J1153 data files not found")


@needs_data
def test_yaml_setup_reproduces_python_setup_chi2_exactly(tmp_path):
    """M13 success criterion: the set-up written to YAML and loaded back gives
    the same mask and the same chi^2, bit for bit, as the same set-up built
    directly in Python (as in scripts/m10d_fit_1A_bulge_disk.py)."""
    import sys

    sys.path.insert(0, sys_path_scripts)
    from m13c_write_example import example_config

    from forwardgal.model import ForwardModel
    from forwardgal.psf import load_psf, make_convolver, make_point_source_renderer
    from forwardgal.quasar import image_fixed_quasar, snap_to_psf
    from forwardgal.regions import fit_region
    from forwardgal.serialize import model_from_dict
    from forwardgal.traced import trace_grid

    c = _common()
    cfg = example_config()
    write_yaml(cfg, tmp_path / "model.yaml")
    s = load(tmp_path / "model.yaml", flux_parameters=False)  # NNLS fluxes, as the M10d script (pre-F5)

    # Python, as in the M10d script
    img = c.load_example_cutout()
    field = c.load_field()
    region = fit_region(c.example_mask(img.data.shape), margin=35)
    r0, _, c0, _ = region.box
    shape = region.box_shape
    grid = trace_grid(img.wcs, shape, field, subsample=5, origin=(r0, c0))
    psf = load_psf(ROOT / c.PSF_FILE)
    yy, xx = np.indices(psf.kernel.shape)
    psf_ref = ((psf.kernel * yy).sum(), (psf.kernel * xx).sum())
    qso = image_fixed_quasar(*c.QSO_1A, field)
    qso_rc, _ = snap_to_psf(qso.grid_position(img.wcs, origin=(r0, c0)), psf_ref, psf.centre)
    tmpl = np.asarray(make_point_source_renderer(psf.kernel, shape, psf_ref)(*qso_rc))
    py = ForwardModel(model_from_dict(cfg["model"]), grid, make_convolver(psf.kernel, shape), region.in_box(img.data),
                      region.in_box(img.sigma), region.in_box(region.mask), region.render_pixels_in_box(),
                      img.pixscale, fixed_templates={"QSO 1A": tmpl})

    assert np.array_equal(s.region.mask, region.mask)
    assert s.region.box == region.box
    u = py.params.start()
    for du in (0.0, 0.003):  # at the start and slightly away from it
        assert float(s.model.loss(u + du)) == float(py.loss(u + du))
