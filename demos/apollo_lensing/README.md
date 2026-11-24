# Lensing demo: Apollo the cat through the COOL J1153 cluster

The same demo as `../bernie_lensing/` (see its README for how it works), with
`apollo.png` as the picture. The scripts live in `../bernie_lensing/`; this
folder holds the picture and the outputs. Run from the repository root:

    python demos/bernie_lensing/bernie_lensing.py --image demos/apollo_lensing/apollo.png --out demos/apollo_lensing/apollo_lensing.gif
    python demos/bernie_lensing/bernie_lensing.py --image demos/apollo_lensing/apollo.png --clean --out demos/apollo_lensing/apollo_lensing_clean.gif
    python demos/bernie_lensing/bernie_lensing.py --image demos/apollo_lensing/apollo.png --image-only --out demos/apollo_lensing/apollo_lensing_image.gif

1. `apollo_lensing.gif`: source plane (caustics, cyan) and image plane
   (critical curves, red), with axes and a legend.
2. `apollo_lensing_clean.gif`: both panels, picture only, same angular scale.
3. `apollo_lensing_image.gif`: the image plane only, picture only.

`cluster_members.png` is the foreground-cluster overlay (it does not depend on
the picture; a copy of `../bernie_lensing/cluster_members.png`). Lay it over
`apollo_lensing_image.gif` at (0, 0) or `apollo_lensing_clean.gif` at
(600, 0).

Apollo is 2" tall in the source plane (`--height`), on the same path as
Bernie (`--start`).
