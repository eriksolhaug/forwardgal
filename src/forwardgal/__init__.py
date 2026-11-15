"""forwardgal: forward modelling of cluster-lensed galaxies with fixed deflection maps."""

import jax

# Use 64-bit floats in JAX (its default is 32-bit, ~7 significant digits),
# needed for accurate flux sums over many pixels and for NUTS sampling.
jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
