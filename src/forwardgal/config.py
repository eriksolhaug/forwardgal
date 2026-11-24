"""A whole fit set-up from one configuration dictionary / YAML file (M13c).

Sections (relative paths are relative to ``base_dir``, normally the folder of
the YAML file; "~" is the home folder)::

    label: bulge-disk                       # used for fit-folder names (M13d)
    data:
      image: {file: ~/.../f115w_i2d.fits, ext: SCI}
      sigma: {file: ~/.../f115w_i2d.fits, ext: ERR}
      centre: [178.3352306, 7.9335254]      # cutout centre, RA, Dec (deg)
      size_arcsec: 9.0
      zeropoint: null                       # null: AB zeropoint from the header
    lens:
      deflection_x: data/lensmodel/bestopt_1.524_dplx.fits
      deflection_y: data/lensmodel/bestopt_1.524_dply.fits
      z_ref: 1.524                          # redshift the maps are made for
      x_points_west: true                   # Lenstool's dplx points west
      interpolation: bicubic                # or bilinear
    cosmology: {H0: 70.0, Om0: 0.3, Ode0: 0.7, z_lens: 0.4301, z_src: 1.524}
    mask:
      polygon_radec: [[ra, dec], ...]       # fit pixels: inside this polygon (deg)
      margin_px: 35                         # render margin around the mask
    psf: {file: data/...PSF.fits, ext: 0, reference: centroid}   # or [row, col]
    render: {subsample: 5, patch: 7, exact_pixels: 200, gl_nodes: 12, bend_split: 4}
    fit: {method: lm, n_starts: 5, seed: 0}
    model: {components: [...], ties: {}, constraints: []}   # forwardgal.serialize

A point component (e.g. the quasar) with a locked image position, given as
RA/Dec (or as locked x0, y0 params in arcsec), optionally with its own PSF::

    - {name: QSO 1A, kind: point, plane: image, ra: 178.3352306, dec: 7.9335254,
       max_snap: 0.1,                       # null = trust the PSF (no limit)
       psf: {file: psf_1A.fits, ext: 0, reference: centroid},   # optional
       params: {}}

Old files with a separate ``quasar:`` section are still read: the section
becomes such a component (``upgrade_config``); it is never written.

Cosmology and redshifts are always given by the user (never read from the
Lenstool files). The deflection maps are rescaled by D_ls/D_s only when
z_src != z_ref.

No lens (GALFIT-like fits, N2): leave out ``lens:`` (or ``lens: null``);
``cosmology:`` is then not needed. theta is measured from the cutout centre
(``data: centre``), or from ``data: reference: [ra, dec]`` if given, with
theta_x = +east, theta_y = +north in arcsec as usual. Every component must
be in the image plane (there is no source plane); points get mu = 1.
Internally the lens is lens.no_lens (alpha = 0, so beta = theta), so all
the rest of the code is the same. ``data: reference`` with a lens is an
error: theta = 0 is then the lens model's reference position.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forwardgal.cosmology import Cosmology
from forwardgal.data import load_cutout
from forwardgal.images import magnification
from forwardgal.lens import load_deflection, no_lens
from forwardgal.model import ForwardModel
from forwardgal.photometry import add_flux_parameters
from forwardgal.psf import load_psf, make_convolver
from forwardgal.quasar import place_point, theta_to_grid
from forwardgal.regions import fit_region, polygon_mask
from forwardgal.render import RenderSettings
from forwardgal.serialize import model_from_dict, read_yaml
from forwardgal.traced import trace_grid

SECTIONS = {"label", "data", "lens", "cosmology", "mask", "psf", "quasar", "render", "fit", "model", "history_dir",
            "seed",  # seed: where the starting values came from (forwardgal.seed; informational)
            "band"}  # band: the band's name for the GUI's tabs, e.g. F150W (optional; see band_name)


@dataclass
class Setup:
    """Everything built from a configuration (the model and its ingredients)."""

    cfg: dict
    model: ForwardModel
    img: object  # forwardgal.data.ImageData (the cutout)
    field: object  # forwardgal.lens.DeflectionField (rescaled to z_src)
    region: object  # forwardgal.regions.FitRegion
    grid: object  # forwardgal.traced.TracedGrid of the render box
    points: dict  # point component name -> PointInfo (e.g. the quasar)
    files: dict  # role -> absolute Path of every input file (for checksums, M13d)
    radec: dict  # point name -> (ra, dec, x0, y0) for points given by RA/Dec (config.restore_radec)
    fluxes_added: list = None  # components given a flux parameter by build (F5), from one linear solve
    lensed: bool = True  # False: no lens (field = lens.no_lens, image plane only)

    @property
    def quasar_beta(self):
        """beta (arcsec) of the first point component, or None (for plots)."""
        return next(iter(self.points.values())).beta if self.points else None


@dataclass(frozen=True)
class PointInfo:
    """Where a locked image-plane point component was placed by ``build``."""

    theta: tuple  # (x0, y0) as given (arcsec, +x east)
    radec: tuple  # its RA, Dec (deg)
    beta: tuple  # traced source position beta = theta - alpha(theta) (arcsec)
    rc: tuple  # snapped 0-based [row, col] on the render box (where the template is drawn)
    snap: float  # how far it was moved onto its PSF stamp's pixel grid (px)
    psf: str  # the PSF file used (its own, or the global one)
    mu: float  # signed point magnification 1 / det A at theta (images.magnification)


def resolve(path, base_dir):
    """Absolute Path: "~" expanded, relative paths taken from base_dir."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path(base_dir) / p).resolve()


def band_name(cfg, base_dir="."):
    """The band of a configuration (the GUI's tab name): ``band:`` if given, else the FILTER keyword
    of the data file's header (primary, then the data extension; JWST: e.g. "F115W"), else the file name."""
    if cfg.get("band"):
        return str(cfg["band"])
    image = cfg.get("data", {}).get("image", {})
    path = resolve(image.get("file", ""), base_dir) if image.get("file") else None
    if path is not None and path.exists():
        from astropy.io import fits

        try:
            for ext in (0, image.get("ext", 0)):
                f = fits.getheader(path, ext).get("FILTER")
                if f:
                    return str(f).strip()
        except (OSError, KeyError, IndexError):
            pass
    return path.stem if path is not None else str(cfg.get("label", "band"))


def absolute_paths(cfg, base_dir):
    """Copy of ``cfg`` with every input-file path (and history_dir) absolute, taken from ``base_dir``.

    For configurations written somewhere else than their YAML's folder (the
    fit history: fits/NNNN_label/model.yaml), so they still find their files.
    Paths: data image and sigma, lens maps, the PSF, points' own PSFs, an old
    quasar: section's PSF, history_dir.
    """
    cfg = copy.deepcopy(cfg)

    def fix(d, key):
        if isinstance(d, dict) and d.get(key):
            d[key] = str(resolve(d[key], base_dir))

    for k in ("image", "sigma"):
        fix(cfg.get("data", {}).get(k), "file")
    for k in ("deflection_x", "deflection_y"):
        fix(cfg.get("lens"), k)
    fix(cfg.get("psf"), "file")
    fix(cfg.get("quasar", {}).get("psf") if cfg.get("quasar") else None, "file")
    for comp in cfg.get("model", {}).get("components", []):
        fix(comp.get("psf"), "file")
    fix(cfg, "history_dir")
    return cfg


def radec_to_pixels(wcs, polygon_radec):
    """(M, 2) [RA, Dec] in degrees -> (M, 2) 0-based [row, col] of the cutout."""
    radec = np.asarray(polygon_radec, dtype=float)
    col, row = wcs.world_to_pixel_values(radec[:, 0], radec[:, 1])  # astropy: 0-based (x = col, y = row)
    return np.column_stack([row, col])


def pixels_to_radec(wcs, polygon_rc):
    """Inverse of ``radec_to_pixels``: (M, 2) [row, col] -> (M, 2) [RA, Dec] in degrees."""
    rc = np.asarray(polygon_rc, dtype=float)
    ra, dec = wcs.pixel_to_world_values(rc[:, 1], rc[:, 0])
    return np.column_stack([ra, dec])


# --- point components given by RA/Dec, and the old ``quasar:`` section (2026-11-22) ---


def upgrade_config(cfg):
    """Copy of ``cfg`` with an old ``quasar:`` section turned into a point component.

    Old files (before 2026-11-22) give the quasar as a section of its own,
        quasar: {name: QSO 1A, mode: image_fixed, ra: ..., dec: ..., max_snap: 0.1}
    which becomes the model component
        {name: QSO 1A, kind: point, plane: image, ra: ..., dec: ..., max_snap: 0.1, params: {}}
    appended LAST to model.components: where the quasar template has always
    been in the fitter's template list, so the amplitude order is unchanged.
    A config without the section is returned as a plain copy.
    """
    cfg = copy.deepcopy(cfg)
    qc = cfg.pop("quasar", None)
    if not qc:
        return cfg
    if qc.get("mode", "image_fixed") != "image_fixed":
        raise NotImplementedError(f"quasar mode {qc['mode']!r} in config files is not supported")
    name = qc.get("name", "QSO")
    if any(c["name"] == name for c in cfg["model"]["components"]):
        raise ValueError(f"the quasar name {name!r} is also a model component name")
    cfg["model"]["components"].append({"name": name, "kind": "point", "plane": "image",
                                       "ra": float(qc["ra"]), "dec": float(qc["dec"]),
                                       "max_snap": qc.get("max_snap", 0.1), "params": {}})
    return cfg


def resolve_points(spec, sky_to_theta):
    """Model spec with RA/Dec point positions converted to locked x0, y0 (arcsec).

    spec : model dict; a point component may give ``ra`` and ``dec`` (deg)
        instead of x0, y0 params: a fixed image position (plane "image").
    sky_to_theta : (ra, dec) -> (theta_x, theta_y) in arcsec, e.g. the lens
        field's own conversion (DeflectionField.sky_to_theta).
    Returns (new spec, record) with record = {name: (ra, dec, x0, y0)} for
    each converted point (``restore_radec`` uses it when saving). The input
    is not changed.
    """
    spec = copy.deepcopy(spec)
    record = {}
    for c in spec["components"]:
        if "ra" not in c and "dec" not in c:
            continue
        name = c["name"]
        if c.get("kind") != "point":
            raise ValueError(f"{name}: only point components take an ra/dec position")
        if "ra" not in c or "dec" not in c:
            raise ValueError(f"{name}: give both ra and dec")
        if c.get("plane") != "image":
            raise ValueError(f"{name}: ra/dec is an image position, so plane must be 'image'")
        if {"x0", "y0"} & set(c.get("params", {})):
            raise ValueError(f"{name}: give the position as ra/dec or as x0/y0, not both")
        ra, dec = float(c.pop("ra")), float(c.pop("dec"))
        tx, ty = (float(t) for t in sky_to_theta(ra, dec))
        c.setdefault("params", {}).update(x0={"value": tx, "free": False}, y0={"value": ty, "free": False})
        record[name] = (ra, dec, tx, ty)
    return spec, record


def restore_radec(spec, record, tol=1e-9):
    """Copy of a resolved spec for saving: unmoved RA/Dec points get ra/dec back.

    A point in ``record`` whose x0, y0 are still locked at the converted
    values (within ``tol`` arcsec) is written with the ra/dec it was given
    (its other parameters, e.g. its flux, are kept); one that was moved or
    unlocked keeps x0, y0.
    """
    spec = copy.deepcopy(spec)
    for i, c in enumerate(spec["components"]):
        if c["name"] not in record or c["kind"] != "point":
            continue
        ra, dec, tx, ty = record[c["name"]]
        p = c["params"]
        pos = {k: p.get(k) for k in ("x0", "y0")}
        unmoved = (all(v is not None and set(v) == {"value", "free"} and v["free"] is False for v in pos.values())
                   and abs(pos["x0"]["value"] - tx) <= tol and abs(pos["y0"]["value"] - ty) <= tol)
        if unmoved:
            rest = {k: v for k, v in c.items() if k not in ("name", "kind", "plane", "params")}
            others = {k: v for k, v in p.items() if k not in ("x0", "y0")}  # e.g. its flux parameter
            spec["components"][i] = {"name": c["name"], "kind": "point", "plane": c["plane"],
                                     "ra": ra, "dec": dec, **rest, "params": others}
    return spec


def build(cfg, base_dir=".", flux_parameters=True):
    """Configuration dictionary -> Setup with a ready-to-fit ForwardModel.

    flux_parameters : True (default, F5): a component without a flux
        parameter gets one, from one linear (NNLS) solve at the starting
        shapes (photometry.add_flux_parameters), so every flux is fitted
        like any other parameter. False: such amplitudes stay solved by NNLS
        in every evaluation (the behaviour before 2026-11-22; to reproduce
        old fits exactly).
    """
    unknown = set(cfg) - SECTIONS
    if unknown:
        raise ValueError(f"config: unknown section(s) {sorted(unknown)}")
    cfg = upgrade_config(cfg)  # an old quasar: section -> a point component (a copy: the input is untouched)
    files = {}

    # Data cutout.
    d = cfg["data"]
    files["image"] = resolve(d["image"]["file"], base_dir)
    files["sigma"] = resolve(d["sigma"]["file"], base_dir)
    img = load_cutout(files["image"], d["image"].get("ext", 0), files["sigma"], d["sigma"].get("ext", 0),
                      *d["centre"], size_arcsec=d["size_arcsec"], zeropoint=d.get("zeropoint"))

    # Lens (rescaled to the source redshift if needed), or none.
    lz = cfg.get("lens")
    lensed = lz is not None
    if lensed:
        if d.get("reference") is not None:
            raise ValueError("data: reference is for fits without a lens; with a lens, theta = 0 is the "
                             "lens model's reference position")
        files["deflection_x"] = resolve(lz["deflection_x"], base_dir)
        files["deflection_y"] = resolve(lz["deflection_y"], base_dir)
        field = load_deflection(files["deflection_x"], files["deflection_y"], z_ref=lz["z_ref"],
                                x_points_west=lz.get("x_points_west", True),
                                interpolation=lz.get("interpolation", "bicubic"))
        c = cfg["cosmology"]
        if c["z_src"] != lz["z_ref"]:
            field = field.rescaled(c["z_lens"], c["z_src"], Cosmology(c["H0"], c["Om0"], c["Ode0"]))
    else:
        field = _no_lens_field(d)
        source = [x["name"] for x in cfg["model"]["components"] if x.get("plane") == "source"]
        if source:
            raise ValueError(f"no lens: there is no source plane; put {source} in the image plane "
                             "(plane: image, x0/y0 = theta in arcsec from the reference)")

    # Mask (sky polygon -> cutout pixels) and the render region.
    m = cfg["mask"]
    mask = polygon_mask(img.data.shape, radec_to_pixels(img.wcs, m["polygon_radec"]))
    if not mask.any():
        raise ValueError("the mask polygon contains no pixels of the cutout")
    region = fit_region(mask, margin=m.get("margin_px", 35))
    r0, _, c0, _ = region.box
    shape = region.box_shape
    settings = RenderSettings(**cfg.get("render", {}))
    grid = trace_grid(img.wcs, shape, field, subsample=settings.subsample, origin=(r0, c0))

    # PSF.
    p = cfg["psf"]
    files["psf"] = resolve(p["file"], base_dir)
    psf = load_psf(files["psf"], ext=p.get("ext", 0))
    # reference: where in the stamp its point lies; the light centroid by default (M6: the PSF of 1A
    # is made for its position). Used for point components (quasar.place_point).

    # Point components (e.g. the quasar): RA/Dec -> theta, then one template each,
    # drawn with the point's own PSF if it names one (else the global PSF).
    cfg["model"], radec = resolve_points(cfg["model"], field.sky_to_theta)
    params = model_from_dict(cfg["model"])
    to_grid = theta_to_grid(img.wcs, field, (r0, c0))
    point_templates, points = {}, {}
    for comp in params.components:
        if comp.kind != "point":
            continue
        x0, y0 = comp.params["x0"], comp.params["y0"]
        tied = {f"{comp.name}.x0", f"{comp.name}.y0"} & set(params.ties)
        if comp.plane != "image" or x0.free or y0.free or tied:
            raise NotImplementedError(f"{comp.name}: only points with a locked image position (plane image, "
                                      "x0 and y0 fixed, or ra/dec) are supported so far")
        if comp.psf is not None:
            files[f"psf:{comp.name}"] = path = resolve(comp.psf["file"], base_dir)
            kernel = load_psf(path, ext=comp.psf.get("ext", 0)).kernel
            ref = comp.psf.get("reference", "centroid")
        else:
            path, kernel, ref = files["psf"], psf.kernel, p.get("reference", "centroid")
        bx, by = field.backtrace(x0.value, y0.value)
        if not np.isfinite(float(bx)):
            raise ValueError(f"{comp.name}: its position lies outside the deflection maps")
        ra, dec = field.theta_to_sky(x0.value, y0.value)
        pp = place_point(to_grid(x0.value, y0.value), kernel, shape, reference=ref, max_snap=comp.max_snap)
        point_templates[comp.name] = pp.template
        points[comp.name] = PointInfo(theta=(x0.value, y0.value), radec=(float(ra), float(dec)),
                                      beta=(float(bx), float(by)), rc=pp.rc, snap=pp.snap, psf=str(path),
                                      mu=magnification(field, x0.value, y0.value))

    def make(p):
        return ForwardModel(p, grid, convolve, region.in_box(img.data), region.in_box(img.sigma),
                            region.in_box(region.mask), region.render_pixels_in_box(), img.pixscale,
                            point_templates=point_templates, settings=settings, zeropoint=img.zeropoint)

    convolve = make_convolver(psf.kernel, shape)
    model, added = make(params), []
    if flux_parameters and any(c.flux_key is None for c in params.components):
        # F5: one linear solve at the starting shapes gives the missing fluxes their starting values.
        _, a = model.solve(params.start())
        amps = dict(zip(model.names, [float(x) for x in a]))
        cfg["model"], added = add_flux_parameters(cfg["model"], amps, img.zeropoint, img.pixscale)
        model = make(model_from_dict(cfg["model"]))
    return Setup(cfg, model, img, field, region, grid, points, files, radec, added, lensed)


def _no_lens_field(data_section):
    """lens.no_lens around data: reference (default: the cutout centre), covering the cutout."""
    ra_c, dec_c = data_section["centre"]
    ra0, dec0 = data_section.get("reference") or (ra_c, dec_c)
    # The cutout lies within size/sqrt(2) of its centre; add the reference's distance from it, and room.
    offset = 3600.0 * np.hypot((ra_c - ra0) * np.cos(np.radians(dec0)), dec_c - dec0)
    half = offset + data_section["size_arcsec"] + 5.0
    return no_lens(ra0, dec0, half_size_arcsec=half)


def load(path, **kwargs):
    """Read a YAML configuration and build it (paths relative to the file's folder; kwargs: build)."""
    path = Path(path)
    return build(read_yaml(path), base_dir=path.parent, **kwargs)
