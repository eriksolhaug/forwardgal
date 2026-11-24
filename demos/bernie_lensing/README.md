# Lensing demo: a picture through the COOL J1153 cluster

`bernie_lensing.py` places a colour picture (RGBA PNG; transparent pixels are
empty sky) in the source plane, moves it from outside the tangential caustic
to the source position of the quasar (traced from image 1A), and writes an
animated GIF. Three versions:

    python demos/bernie_lensing/bernie_lensing.py               # bernie_lensing.gif
    python demos/bernie_lensing/bernie_lensing.py --clean       # bernie_lensing_clean.gif
    python demos/bernie_lensing/bernie_lensing.py --image-only  # bernie_lensing_image.gif

1. Source plane with the caustics (cyan) next to the image plane with the
   critical curves (red), with axes and a legend.
2. `--clean`: the same two panels, picture only on black (no axes, curves,
   titles or legend), both with the same field of view (arcsec per pixel), so
   sizes in the two planes compare 1-to-1 (`--same-scale`, also available
   for version 1).
3. `--image-only`: just the image plane, picture only on black.

It uses forwardgal's deflection maps (`data/lensmodel/bestopt_1.524_dpl{x,y}.fits`,
z = 1.524, no rescaling), `critical_curves`/`caustics` and
`image_fixed_quasar`. Lensing conserves surface brightness, so each image-plane
pixel shows the picture's colour at β(θ) = θ − α(θ).

Run from the repository root (commands above). Options: `--image` (default `bernie.png` here), `--height` (picture height in the
source plane, arcsec; default 2.0), `--start DX DY` (start offset from the
quasar's β, arcsec east/north; default 5.5 4.5), `--frames`, `--fps`,
`--pixels`, `--ss` (supersampling), `--fov` (image-plane field, arcsec),
`--out`.

Each version takes about 1 to 1.5 minutes and 1.2 GB of memory with the defaults.

## Foreground cluster members (still PNG overlay)

    python demos/bernie_lensing/cluster_members.py [--check]

writes `cluster_members.png`: 600 x 600 RGBA, exactly the image-plane panel of
the GIFs (same `--fov`/`--pixels`; north up, east left), showing only the
cluster members from the JWST colour image (red F444W, green F150W, blue
F115W); everything else is transparent. Lay it over `bernie_lensing_image.gif`
(at 0, 0) or the right half of `bernie_lensing_clean.gif` (at x = 600, y = 0);
both line up pixel for pixel.

Members are the positions in the Lenstool model's `cats/1153ground.cat` plus
the individually optimized galaxies O2-O5 (`bestopt.par`). Detected sources
containing a member are kept; lensed light (quasar images, host arcs) is cut
by colour (it is much redder in F444W than the members), so two small members
embedded in the 1B host light are lost. `--check` writes
`cluster_members_check.png` with the kept regions outlined.
