"""Write data minus (part of) the model as a FITS image (M1, 2026-11-22).

E.g. the data with the quasar (a PSF) subtracted: render only the components
to subtract, then

    hdu = data_minus_model_hdu(img, region, model_box, subtracted, area="box")

img : forwardgal.data.ImageData (the cutout: data, wcs, bunit, zeropoint)
region : forwardgal.regions.FitRegion; region.box = (r0, r1, c0, c1) is the
    render box as the slice [r0:r1, c0:c1] of the cutout (0-based)
model_box : (r1 - r0, c1 - c0) array, the PSF-convolved model of the
    subtracted components on the render box (data units)
area : "box" (the render box only: the model is computed everywhere in it)
    or "cutout" (the whole cutout; outside the render box nothing is
    subtracted, since the model is only computed inside it; the header
    says where the box is)

The image is in the data's units (BUNIT copied) with a celestial WCS for
exactly the pixels written (for "box": the cutout's WCS with CRPIX shifted
by the box corner). Header keywords list the subtracted components.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits


def data_minus_model_hdu(img, region, model_box, subtracted, area="box"):
    """PrimaryHDU of data - model; ``subtracted``: the names of the components in the model."""
    if area not in ("box", "cutout"):
        raise ValueError(f"area must be 'box' or 'cutout', got {area!r}")
    r0, r1, c0, c1 = region.box
    model_box = np.asarray(model_box, float)
    if model_box.shape != (r1 - r0, c1 - c0):
        raise ValueError(f"model_box has shape {model_box.shape}, the render box is {(r1 - r0, c1 - c0)}")
    if area == "box":
        out = np.asarray(img.data, float)[r0:r1, c0:c1] - model_box
        wcs = img.wcs[r0:r1, c0:c1]  # astropy slices [row, col] and shifts CRPIX by the corner
    else:
        out = np.array(img.data, float)
        out[r0:r1, c0:c1] -= model_box
        wcs = img.wcs
    hdr = wcs.to_header()
    hdr["BUNIT"] = img.bunit
    hdr["CONTENT"] = ("data - model", "forwardgal: data minus the SUBTRn components")
    hdr["NSUBTR"] = (len(subtracted), "number of model components subtracted")
    for k, name in enumerate(subtracted, start=1):
        hdr[f"SUBTR{k}"] = (str(name), "subtracted component (PSF-convolved model)")
    hdr["AREA"] = (area, "box: render box only; cutout: whole cutout")
    if area == "cutout":  # 1-based FITS pixel ranges, inclusive
        hdr["BOXROWS"] = (f"{r0 + 1}:{r1}", "rows with the model subtracted (1-based)")
        hdr["BOXCOLS"] = (f"{c0 + 1}:{c1}", "columns with the model subtracted")
    if img.zeropoint is not None:
        hdr["MAGZP"] = (float(img.zeropoint), "AB: m = -2.5 log10(sum of pixels) + MAGZP")
    return fits.PrimaryHDU(out.astype(np.float32), header=hdr)


# --- the image block of a fit (GALFIT-style) ---------------------------------------------

# parameter name -> keyword part and units (GALFIT writes e.g. 1_XC; ours: 1_X0)
KEYWORDS = {"x0": ("X0", "arcsec"), "y0": ("Y0", "arcsec"), "R_e": ("RE", "arcsec"), "n": ("N", ""),
            "e1": ("E1", ""), "e2": ("E2", ""), "q": ("Q", "b/a"), "pa_deg": ("PA", "deg E of N"),
            "mag": ("MAG", "AB mag"), "flux": ("FLUX", "data units x pixels")}


def _key(k, name):
    """FITS keyword of parameter ``name`` of component number k (<= 8 characters; else HIERARCH)."""
    part = KEYWORDS.get(name, (name.upper(), ""))[0]
    return f"{k}_{part}"


def imgblock(setup, result, laplace=None, fit_name=None, region_number=1):
    """HDUList of a fit's image block, in GALFIT's order (imgblock.fits).

        0  primary: no image; a summary header (fit, chi2, dof, N, BIC, components)
        1  DATA      the data in the render box          (BUNIT of the data)
        2  MODEL     the best-fit model (PSF-convolved)  + the best-fit values in its header
        3  RESIDUAL  data - model                        (data units; not divided by sigma)
        4  SIGMA     the 1-sigma noise
        5  MASK      1 = fit pixel, 0 = not (uint8)

    Each image extension has EXTNAME and EXTVER = ``region_number`` (a later fit
    of several image regions writes DATA,1 DATA,2 ...: astropy finds them as
    hdul["DATA", 2]) and the celestial WCS of the render box (the cutout's
    WCS with CRPIX shifted by the box corner). Positions in the MODEL header
    are theta (image plane) or beta (source plane) in arcsec from the
    reference, x east, y north, as everywhere in forwardgal; the comments
    give the Laplace errors when ``laplace`` is given.
    """
    import jax.numpy as jnp

    from forwardgal.history import bic

    model = setup.model
    u = jnp.asarray(result.u)
    data = np.asarray(model.data, float)
    mdl = np.asarray(model.model_image(u), float)
    sigma = np.asarray(model.sigma, float)
    mask = np.asarray(model.mask, bool)
    c2, dof = (float(x) for x in model.chi2_dof(u))
    n_pix = int(mask.sum())
    r0, r1, c0, c1 = setup.region.box
    wcs_hdr = setup.img.wcs[r0:r1, c0:c1].to_header()

    prim = fits.PrimaryHDU()
    h = prim.header
    h["ORIGIN"] = "forwardgal"
    if fit_name:
        h["FITNAME"] = (str(fit_name), "fit folder")
    h["LABEL"] = str(setup.cfg.get("label", ""))
    h["CHI2"] = (c2, "chi^2 in the mask")
    h["DOF"] = (int(dof), "nu = N - fitted quantities")
    h["CHI2NU"] = (c2 / dof, "chi^2 / nu")
    h["NPIX"] = (n_pix, "fit pixels N (MASK = 1)")
    b = bic(c2, dof, n_pix)
    h["BIC"] = (b, "chi^2 + k ln N, k = N - nu")
    h["LENSED"] = (bool(setup.lensed), "False: no lens model (beta = theta)")
    if setup.img.zeropoint is not None:
        h["MAGZP"] = (float(setup.img.zeropoint), "AB: m = -2.5 log10(sum of pixels) + MAGZP")
    comps = model.params.components
    h["NCOMP"] = (len(comps), "model components (see MODEL header)")
    for k, c in enumerate(comps, start=1):
        h[f"COMP{k}"] = (c.name, f"{c.kind}, {c.plane} plane")

    def image(arr, name, unit=True):
        hdu = fits.ImageHDU(arr, header=wcs_hdr.copy(), name=name, ver=region_number)
        if unit:
            hdu.header["BUNIT"] = setup.img.bunit
        hdu.header["REGION"] = (region_number, "image region (EXTVER)")
        return hdu

    hdus = [prim, image(data.astype(np.float32), "DATA"), image(mdl.astype(np.float32), "MODEL"),
            image((data - mdl).astype(np.float32), "RESIDUAL"), image(sigma.astype(np.float32), "SIGMA"),
            image(mask.astype(np.uint8), "MASK", unit=False)]

    mh = hdus[2].header  # the best-fit values, GALFIT-like
    sig = dict(zip(laplace.names, laplace.sigma)) if laplace is not None else {}
    sig.update({n.removesuffix(" (derived)"): s for n, _, s in (laplace.derived or [])} if laplace is not None
               else {})  # (component names may contain spaces: "QSO 1A.mag (derived)")
    for k, c in enumerate(comps, start=1):
        mh[f"{k}_NAME"] = (c.name, f"{c.kind}, {c.plane} plane")
        vals = result.values.get(c.name, {})
        for pname, v in vals.items():
            unit = KEYWORDS.get(pname, ("", ""))[1]
            s = sig.get(f"{c.name}.{pname}")
            note = (f"+/- {s:.3g} " if s is not None else "") + unit
            key = _key(k, pname)
            mh[key if len(key) <= 8 else f"HIERARCH {key}"] = (float(v), note.strip())
        a = result.amplitudes.get(c.name)
        if a is not None:
            mh[f"{k}_AMP"] = (float(a), "amplitude (flux, data units x pixels)")
    return fits.HDUList(hdus)
