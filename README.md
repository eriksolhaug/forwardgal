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

- **Near each Sérsic centre** (a 7 × 7 patch of pixels by default), the pixel
  integral is computed exactly: in elliptical polar coordinates around the
  centre, the radial integral of a Sérsic profile has a closed form
  (incomplete gamma function). The remaining angular integral is split at the
  pixel's corner directions into smooth pieces, each integrated with
  12-point Gauss–Legendre quadrature. Accuracy: ~10⁻⁶ for any Sérsic index
  0.5 ≤ n ≤ 8. Uniform point sampling is not reliable there: for n ≥ 4 the
  centre pixel is wrong by 4–15% when the profile centre falls between
  sample points, and by large factors (×7 for n = 4, ×10⁴ for n = 8) when it
  falls on one; it depends on where the centre sits, so the model would jump
  as a fit moves it. Next to the centre the error is ~1% (5 × 5 samples,
  R_e = 0.5–1.7 px).
- **Elsewhere**, each pixel is subsampled uniformly (s × s = 5 × 5 points by
  default) and averaged. Measured error: ≲ 1.5 × 10⁻³ of the pixel value for
  the most compact profiles (R_e ≈ 1 px), smaller for larger R_e.
- **Exact integration everywhere** is ~500× slower, so it is used only as a
  check: after every fit, the best-fit model is rendered both ways, both are
  PSF-convolved, and max |hybrid − exact|/σ in the mask and
  Δχ² = χ²_hybrid − χ²_exact are saved with the fit. A warning is issued if
  max > 0.1σ or |Δχ²| > 1; then increase s or the patch size.
- **With lensing**, image pixels are traced back to the source plane once;
  each pixel's footprint there is the quadrilateral through its four traced
  corners. The exactly integrated pixels are the N = 200 pixels (default)
  whose traced centres lie nearest each component's centre *in the source
  plane*. In the image plane these form an ellipse stretched along the arc by
  the local magnification ratio, around every image of the centre. A square
  patch is not enough in a stretched arc: a pixel d pixels along a 10×
  stretched arc is only d/10 source pixels from the core. Measured on a 10×
  stretched image: < 6 × 10⁻⁵ of the peak (a 7 × 7 square gave up to
  1.7 × 10⁻³). Using the quadrilateral as the footprint assumes the
  magnification is nearly constant across one pixel; masks are drawn away
  from critical curves, where it is not.

The rendering settings (subsampling s, patch size, Gauss–Legendre nodes) are
recorded with every saved fit.

## Conventions

- Image plane θ, source plane β = θ − α(θ), all in arcsec.
- θ_x points east (+RA), θ_y points north (+Dec), relative to the lens-model
  reference position.
- Position angles are measured east of north (GALFIT convention).
- Arrays are indexed `[row, col]` = `[y, x]`, 0-based.
