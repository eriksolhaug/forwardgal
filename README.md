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

## How pixels are rendered

A detector pixel records the integral of the surface brightness over its area,
so the model computes pixel-integrated values, not point samples:

- **Near each Sérsic centre** (a small patch of pixels), the pixel integral is
  computed exactly: in elliptical polar coordinates around the centre, the
  radial integral of a Sérsic profile has a closed form (incomplete gamma
  function), leaving a smooth 1-D integral over angle. This is accurate to
  ~10⁻⁶ for any Sérsic index 0.5 ≤ n ≤ 8. Uniform point sampling is not:
  for n ≥ 4 it can be wrong by factors of several in the central pixels.
- **Elsewhere**, where the profile is smooth, each pixel is subsampled
  uniformly (s × s points) and averaged.
- **With lensing** (planned, M4), a pixel's footprint in the source plane is
  approximated by a parallelogram from the local lens Jacobian ∂β/∂θ. This
  approximation is tested explicitly and breaks down very close to critical
  curves.

The rendering settings (subsampling s, patch size, number of angles) are
recorded with every saved fit.

## Conventions

- Image plane θ, source plane β = θ − α(θ), all in arcsec.
- θ_x points east (+RA), θ_y points north (+Dec), relative to the lens-model
  reference position.
- Position angles are measured east of north (GALFIT convention).
- Arrays are indexed `[row, col]` = `[y, x]`, 0-based.
