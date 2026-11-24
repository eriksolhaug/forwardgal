"""M1: components left out of the live model ("off"), and data - model written as FITS."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import forwardgal  # noqa: F401
from forwardgal.export import data_minus_model_hdu
from test_config import needs_data


def fake_image():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [178.3352306, 7.9335254]
    w.wcs.crpix = [26.0, 21.0]  # 1-based
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    data = np.arange(40 * 50, dtype=float).reshape(40, 50)  # [row, col]
    return SimpleNamespace(data=data, wcs=w, bunit="MJy/sr", zeropoint=28.087)


def test_box_area_has_the_box_pixels_and_a_shifted_wcs(tmp_path):
    img, region = fake_image(), SimpleNamespace(box=(5, 25, 10, 40))  # rows 5:25, cols 10:40
    model = np.full((20, 30), 2.0)
    hdu = data_minus_model_hdu(img, region, model, ["QSO 1A"], area="box")
    hdu.writeto(tmp_path / "d.fits")
    with fits.open(tmp_path / "d.fits") as h:
        out, hdr = h[0].data.astype(float), h[0].header
    np.testing.assert_array_equal(out, img.data[5:25, 10:40] - 2.0)
    assert hdr["BUNIT"] == "MJy/sr" and hdr["NSUBTR"] == 1 and hdr["SUBTR1"] == "QSO 1A"
    assert hdr["MAGZP"] == pytest.approx(28.087) and hdr["AREA"] == "box"
    # The box pixel [0, 0] is cutout pixel [5, 10]: the same sky position.
    box_wcs = WCS(hdr)
    np.testing.assert_allclose(box_wcs.wcs_pix2world(0, 0, 0), img.wcs.wcs_pix2world(10, 5, 0), atol=1e-12)
    np.testing.assert_allclose(box_wcs.wcs_pix2world(29, 19, 0), img.wcs.wcs_pix2world(39, 24, 0), atol=1e-12)


def test_cutout_area_subtracts_only_inside_the_box():
    img, region = fake_image(), SimpleNamespace(box=(5, 25, 10, 40))
    hdu = data_minus_model_hdu(img, region, np.ones((20, 30)), ["a", "b"], area="cutout")
    out = hdu.data.astype(float)
    assert out.shape == img.data.shape
    np.testing.assert_array_equal(out[5:25, 10:40], img.data[5:25, 10:40] - 1.0)
    out[5:25, 10:40] = img.data[5:25, 10:40]
    np.testing.assert_array_equal(out, img.data)  # untouched outside the box
    assert hdu.header["BOXROWS"] == "6:25" and hdu.header["BOXCOLS"] == "11:40"  # 1-based, inclusive
    assert hdu.header["SUBTR2"] == "b"


def test_bad_inputs():
    img, region = fake_image(), SimpleNamespace(box=(5, 25, 10, 40))
    with pytest.raises(ValueError, match="area"):
        data_minus_model_hdu(img, region, np.ones((20, 30)), [], area="all")
    with pytest.raises(ValueError, match="render box"):
        data_minus_model_hdu(img, region, np.ones((20, 31)), [])


def test_off_components_are_left_out_of_render_and_fit_fluxes():
    """An "off" component contributes nothing to the model, and Fit fluxes treats it as absent."""
    from test_fit import TRUTH, setup

    from forwardgal.live import LiveModel
    from forwardgal.serialize import model_to_dict

    model, _ = setup(dict(dx=0.004, **TRUTH))  # synthetic: a lensed Sersic "host" + sky, amplitudes by NNLS
    live = LiveModel(SimpleNamespace(model=model, cfg={"model": model_to_dict(model.params)}))
    both = live.render()
    live.off = {"host"}
    r = live.render()
    T = np.asarray(live.template("sky"))
    assert r.amplitudes["host"] == 0.0
    np.testing.assert_allclose(r.model, r.amplitudes["sky"] * T, rtol=1e-12)  # only the sky is left
    m = live.mask  # the sky alone, fitted without the host: its least-squares level
    w = 1 / live.sigma[m] ** 2
    assert r.amplitudes["sky"] == pytest.approx(np.sum(w * T[m] * live.data[m]) / np.sum(w * T[m] ** 2), rel=1e-6)
    assert r.chi2 > both.chi2  # the host's light is now in the residual
    assert set(live.best_fluxes()) == {"sky"}  # never solved while off
    live.off = set()
    assert live.render().chi2 == pytest.approx(both.chi2, rel=1e-12)


@needs_data
def test_imgblock_of_a_fit(tmp_path):
    """imgblock: DATA, MODEL, RESIDUAL, SIGMA, MASK (GALFIT order), EXTVER = region, box WCS, values in MODEL."""
    import sys

    import jax.numpy as jnp
    from astropy.wcs import WCS

    from forwardgal.config import build
    from forwardgal.export import imgblock
    from forwardgal.fit import fit, laplace_errors
    from test_config import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from m13c_write_example import example_config

    s = build(example_config())
    res = fit(s.model, n_starts=1, max_iter=3, progress=False)
    lap = laplace_errors(s.model, res)
    imgblock(s, res, laplace=lap, fit_name="0007_test").writeto(tmp_path / "imgblock.fits")
    with fits.open(tmp_path / "imgblock.fits") as h:
        assert [x.name for x in h] == ["PRIMARY", "DATA", "MODEL", "RESIDUAL", "SIGMA", "MASK"]
        assert h[0].data is None and h[0].header["FITNAME"] == "0007_test"
        assert h["DATA", 1].header["EXTVER"] == 1 and h["MODEL", 1] is h[2]  # found by name and region
        data, mdl, res_img = (h[k].data.astype(float) for k in ("DATA", "MODEL", "RESIDUAL"))
        np.testing.assert_allclose(data, np.asarray(s.model.data), rtol=1e-6)  # (float32 in the file)
        np.testing.assert_allclose(mdl, np.asarray(s.model.model_image(jnp.asarray(res.u))), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(res_img, data - mdl, atol=1e-5)
        mask = h["MASK"].data.astype(bool)
        np.testing.assert_array_equal(mask, np.asarray(s.model.mask))
        c2, dof = s.model.chi2_dof(jnp.asarray(res.u))
        assert h[0].header["CHI2"] == pytest.approx(float(c2)) and h[0].header["NPIX"] == int(mask.sum())
        assert h[0].header["BIC"] == pytest.approx(float(c2) + (mask.sum() - int(dof)) * np.log(mask.sum()))
        # The box WCS: box pixel [0, 0] is cutout pixel [r0, c0].
        r0, _, c0, _ = s.region.box
        np.testing.assert_allclose(WCS(h["DATA"].header).wcs_pix2world(0, 0, 0), s.img.wcs.wcs_pix2world(c0, r0, 0),
                                   atol=1e-10)
        m = h["MODEL"].header
        names = [m[f"{k}_NAME"] for k in range(1, h[0].header["NCOMP"] + 1)]
        k = names.index("disk") + 1
        assert m[f"{k}_RE"] == pytest.approx(res.values["disk"]["R_e"]) and "arcsec" in m.comments[f"{k}_RE"]
        assert m.comments[f"{k}_RE"].startswith("+/- ")  # its Laplace error
        assert f"{k}_PA" in m and f"{k}_MAG" in m  # derived PA (from e1, e2) and the flux parameter
        kq = names.index("QSO 1A") + 1
        assert m.comments[f"{kq}_MAG"].startswith("+/- ")
