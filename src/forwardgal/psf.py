"""Point-spread function: loading (GALFIT conventions) and FFT convolution (JAX).

Conventions
-----------
- The PSF image is normalized to unit sum on load (GALFIT does the same).
- Its centre is pixel [N_rows // 2, N_cols // 2] (0-based): the geometric
  centre for odd N, and pixel N/2 + 1 (1-based) for even N, as in GALFIT.
  A model point source at pixel (r, c) puts the PSF centre on (r, c).
- The PSF must have the image's pixel scale and orientation (here: made by
  the user for this mosaic, north up, east left).

Convolution
-----------
By the convolution theorem, image * PSF = IFFT( FFT(image) x FFT(PSF) ). FFTs
treat arrays as periodic, so both are zero-padded to (box + PSF - 1) before
transforming and the result is cropped back to the box: an ordinary linear
convolution in which light leaving the box is lost and nothing wraps around.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from scipy.fft import next_fast_len


@dataclass(frozen=True)
class PSF:
    """A normalized PSF image.

    Attributes
    ----------
    kernel : (ky, kx) float64 array
        PSF, sum = 1.
    centre : (int, int)
        0-based [row, col] of the PSF centre, (ky // 2, kx // 2).
    raw_sum : float
        Sum of the file's pixel values before normalization.
    source : str
        "path[ext]" of the file it came from.
    """

    kernel: np.ndarray
    centre: tuple
    raw_sum: float
    source: str

    def describe(self):
        ky, kx = self.kernel.shape
        peak = np.unravel_index(np.argmax(self.kernel), self.kernel.shape)
        return "\n".join([
            f"PSF file           : {self.source}",
            f"size               : {ky} x {kx} px, centre [row, col] = {list(self.centre)} (0-based)",
            f"peak pixel         : {[int(p) for p in peak]}",
            f"sum before normalization: {self.raw_sum:.6g} (now 1)",
            f"min value          : {self.kernel.min():.3g} (relative to peak {self.kernel.min() / self.kernel.max():.2g})",
        ])


def load_psf(path, ext=0):
    """Load a PSF image and normalize it to unit sum.

    Warns if the brightest pixel is not at the GALFIT centre (model positions
    would then be offset by the difference), or if there are significantly
    negative pixels. Raises ValueError for NaNs or a non-positive sum.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[ext].data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{path}[{ext}] is not a 2-D image")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}[{ext}] contains NaN or inf")
    total = float(data.sum())
    if total <= 0:
        raise ValueError(f"{path}[{ext}] has non-positive sum {total}")

    kernel = data / total
    centre = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    peak = np.unravel_index(np.argmax(kernel), kernel.shape)
    if tuple(int(p) for p in peak) != centre:
        warnings.warn(
            f"PSF peak at {[int(p) for p in peak]} but GALFIT centre is {list(centre)}: "
            "model positions will be offset by the difference.",
            stacklevel=2,
        )
    if kernel.min() < -1e-3 * kernel.max():
        warnings.warn(f"PSF has negative pixels down to {kernel.min() / kernel.max():.2g} of the peak", stacklevel=2)
    return PSF(kernel=kernel, centre=centre, raw_sum=total, source=f"{path}[{ext}]")


def make_convolver(kernel, shape):
    """Return a JAX function that convolves (ny, nx) images with ``kernel``.

    The kernel's FFT is computed once here, for this image shape, and reused.

    Parameters
    ----------
    kernel : (ky, kx) array
        Normalized PSF with its centre at [ky // 2, kx // 2].
    shape : (ny, nx)
        Shape of the images to convolve (e.g. a FitRegion render box).

    Returns
    -------
    convolve : callable
        image (ny, nx) -> PSF-convolved image (ny, nx), same units.
    """
    ky, kx = kernel.shape
    ny, nx = shape
    # Padded size >= ny + ky - 1 (no wrap-around), rounded up to a size whose
    # FFT is fast (a product of small primes).
    py, px = next_fast_len(ny + ky - 1), next_fast_len(nx + kx - 1)
    kernel_ft = jnp.fft.rfft2(jnp.asarray(kernel), s=(py, px))
    cy, cx = ky // 2, kx // 2  # crop offset: the PSF centre

    def convolve(image):
        full = jnp.fft.irfft2(jnp.fft.rfft2(image, s=(py, px)) * kernel_ft, s=(py, px))
        return full[cy : cy + ny, cx : cx + nx]

    return convolve


def fourier_shift(image, dy, dx):
    """Shift an image by (dy, dx) pixels (rows, cols) with the Fourier shift theorem.

    out(r, c) = image(r - dy, c - dx), i.e. features move by +dy rows, +dx cols.
    For an EVEN image dimension the Nyquist coefficient cannot carry the
    shift (see ``_odd_fast_len``); the image is then zero-padded by one
    pixel to an odd size internally and cropped back.
    Exact for band-limited (Nyquist-sampled) images; for undersampled images
    it rings (measured in M6: up to ~10 sigma at a half-pixel shift for a
    strongly undersampled PSF at the quasar peak, <~1 sigma for a PSF like the
    F115W one, ~0 for a well-sampled one). The image is treated as periodic,
    so its edges must be ~0 (true for PSF stamps). Flux is conserved exactly;
    differentiable in dy, dx.
    """
    ny, nx = image.shape
    py, px = ny + (ny % 2 == 0), nx + (nx % 2 == 0)  # odd sizes
    ky = jnp.fft.fftfreq(py)[:, None]
    kx = jnp.fft.rfftfreq(px)[None, :]
    phase = jnp.exp(-2j * jnp.pi * (ky * dy + kx * dx))
    return jnp.fft.irfft2(jnp.fft.rfft2(image, s=(py, px)) * phase, s=(py, px))[:ny, :nx]


def _odd_fast_len(n):
    """Smallest ODD length >= n whose FFT is fast (a product of small primes).

    Sub-pixel (Fourier) shifts need odd FFT lengths to be pure shifts: an
    even length has a single Nyquist-frequency coefficient that a real FFT
    must keep real, so part of a complex phase shift is lost there.

    (Independently of this, ANY Fourier shift of a PSF whose core is not
    band-limited leaves sinc tails ~ sin(pi d) / (pi x) that alternate in
    sign along the rows/columns through the core; for the F115W PSF a shift
    of d = 0.03 px gives ~1e-4 of the peak. Avoid shifting a PSF made for the
    exact position: see ``quasar.snap_to_psf``.)
    """
    m = next_fast_len(n)
    while m % 2 == 0 or next_fast_len(m) != m:
        m = next_fast_len(m + 1)
    return m


def make_point_source_renderer(kernel, shape, psf_ref=None):
    """Return a JAX function that renders a unit-flux point source on a (ny, nx) grid.

    The point source at fractional pixel (row, col) of the grid is the PSF
    placed so that its reference point ``psf_ref`` (stamp coordinates) lands
    on (row, col). Computed by FFT: a delta function at (row, col) is a phase
    ramp in Fourier space, multiplied by the PSF's transform (zero-padded, so
    nothing wraps around). Placements at whole-pixel offsets reproduce the
    stamp exactly; fractional offsets are Fourier shifts (exact for a
    Nyquist-sampled PSF).

    Parameters
    ----------
    kernel : (ky, kx) array
        Normalized PSF (sum 1), centre at [ky // 2, kx // 2].
    shape : (ny, nx)
        Grid shape (e.g. a FitRegion render box).
    psf_ref : (float, float), optional
        Stamp position (0-based [row, col]) of the point source the PSF was
        made for. Default: the stamp centre (GALFIT convention). For a PSF
        made for a specific quasar position, use where that position lies in
        the stamp (for the COOL J1153 F115W PSF: its light centroid).

    Returns
    -------
    render : callable
        (row, col) -> (ny, nx) image of a unit-flux point source (pixel values
        sum to 1 if the PSF lies fully inside the grid).
    """
    ky, kx = kernel.shape
    ny, nx = shape
    cy, cx = ky // 2, kx // 2
    ref_r, ref_c = (cy, cx) if psf_ref is None else psf_ref
    py, px = _odd_fast_len(ny + ky - 1), _odd_fast_len(nx + kx - 1)  # odd: see _odd_fast_len
    kernel_ft = jnp.fft.rfft2(jnp.asarray(kernel), s=(py, px))
    fy = jnp.fft.fftfreq(py)[:, None]
    fx = jnp.fft.rfftfreq(px)[None, :]

    def render(row, col):
        # The stamp centre must go to (row, col) - (ref - centre); in the padded,
        # cropped convolution frame a delta at grid position p is a delta at p.
        dr = row - (ref_r - cy)
        dc = col - (ref_c - cx)
        phase = jnp.exp(-2j * jnp.pi * (fy * dr + fx * dc))
        full = jnp.fft.irfft2(kernel_ft * phase, s=(py, px))
        return full[cy : cy + ny, cx : cx + nx]

    return render
