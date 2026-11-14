# forwardgal

Forward modelling of strongly lensed galaxies behind galaxy clusters.

`forwardgal` fits parametric source-plane models (Sérsic components, Fourier
isophote modes, and a point source for a quasar) to imaging data, using a
**fixed** lens model supplied as deflection maps (e.g. from Lenstool). Each
image-plane pixel is ray-traced to the source plane with β = θ − α(θ), the
source is evaluated there, and the result is convolved with the PSF. The fit
is an image-plane χ² inside a mask, with gradient-based optimization and NUTS
sampling (JAX + numpyro).

Status: early development. See the milestones for the build plan.

## Installation (development)

```bash
conda create -n forwardgal python=3.12
conda activate forwardgal
pip install -e ".[test]"
```

## Conventions

- Image plane θ, source plane β = θ − α(θ), all in arcsec.
- θ_x points east (+RA), θ_y points north (+Dec), relative to the lens-model
  reference position.
- Position angles are measured east of north (GALFIT convention).
- Arrays are indexed `[row, col]` = `[y, x]`, 0-based.
