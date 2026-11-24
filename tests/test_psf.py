import warnings

import jax
import numpy as np
import pytest
from astropy.io import fits
from scipy.signal import fftconvolve

import forwardgal  # noqa: F401
from forwardgal.psf import load_psf, make_convolver


def write(path, data):
    fits.PrimaryHDU(np.asarray(data, np.float32)).writeto(path)
    return path


def gaussian_psf(n=21, sigma=2.0, dy=0.0, dx=0.0):
    y, x = np.mgrid[0:n, 0:n] - n // 2
    return np.exp(-((x - dx) ** 2 + (y - dy) ** 2) / (2 * sigma**2))


def test_load_normalizes_and_finds_centre(tmp_path):
    psf = load_psf(write(tmp_path / "p.fits", 3.0 * gaussian_psf(21)))
    assert psf.kernel.sum() == pytest.approx(1.0, rel=1e-12)
    assert psf.centre == (10, 10)
    assert psf.raw_sum == pytest.approx(3.0 * gaussian_psf(21).sum(), rel=1e-6)
    assert "21 x 21" in psf.describe()


def test_even_size_centre_follows_galfit(tmp_path):
    # Even N: GALFIT centre is pixel N/2 + 1 (1-based) = N/2 (0-based).
    psf = load_psf(write(tmp_path / "p.fits", gaussian_psf(21)[:20, :20]))
    assert psf.centre == (10, 10)


def test_offcentre_peak_warns(tmp_path):
    with pytest.warns(UserWarning, match="GALFIT centre"):
        load_psf(write(tmp_path / "p.fits", gaussian_psf(21, dy=2.0)))


def test_nan_rejected(tmp_path):
    data = gaussian_psf(21)
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        load_psf(write(tmp_path / "p.fits", data))


def test_delta_function_reproduces_psf():
    # A single bright pixel at [r, c] becomes the PSF centred on [r, c].
    k = gaussian_psf(15, sigma=1.5, dx=0.7)  # asymmetric, to catch flips
    k /= k.sum()
    img = np.zeros((40, 50))
    img[17, 23] = 1.0
    out = np.asarray(make_convolver(k, img.shape)(img))
    np.testing.assert_allclose(out[17 - 7 : 17 + 8, 23 - 7 : 23 + 8], k, atol=1e-14)


def test_matches_scipy_and_conserves_flux():
    rng = np.random.default_rng(0)
    k = gaussian_psf(21, sigma=2.5, dy=0.4)
    k /= k.sum()
    img = np.zeros((60, 45))
    img[20:40, 15:30] = rng.uniform(0, 1, (20, 15))  # well inside: no light lost
    out = np.asarray(make_convolver(k, img.shape)(img))
    np.testing.assert_allclose(out, fftconvolve(img, k, mode="same"), atol=1e-12)
    assert out.sum() == pytest.approx(img.sum(), rel=1e-12)


def test_no_wraparound():
    # Light at the left edge must not appear at the right edge.
    k = gaussian_psf(21, sigma=3.0)
    k /= k.sum()
    img = np.zeros((30, 30))
    img[15, 0] = 1.0
    out = np.asarray(make_convolver(k, img.shape)(img))
    assert np.abs(out[:, -5:]).max() < 1e-15


def test_differentiable():
    k = gaussian_psf(11)
    k /= k.sum()
    conv = make_convolver(k, (20, 20))
    g = jax.grad(lambda a: conv(a * np.ones((20, 20))).sum())(2.0)
    assert np.isfinite(g)


# --- Fourier shift --------------------------------------------------------------

from forwardgal.psf import fourier_shift  # noqa: E402


@pytest.mark.parametrize("dy, dx", [(0.3, -0.45), (1.0, 2.0), (-0.5, 0.0)])
def test_fourier_shift_matches_analytic_gaussian(dy, dx):
    # A well-sampled Gaussian (sigma = 2.5 px) shifted analytically.
    base = gaussian_psf(41, sigma=2.5)
    shifted = np.asarray(fourier_shift(base, dy, dx))
    np.testing.assert_allclose(shifted, gaussian_psf(41, sigma=2.5, dy=dy, dx=dx), atol=1e-10)


def test_fourier_shift_conserves_flux_and_is_differentiable():
    base = gaussian_psf(31, sigma=2.0)
    assert float(fourier_shift(base, 0.37, -0.21).sum()) == pytest.approx(base.sum(), rel=1e-12)
    g = jax.grad(lambda d: fourier_shift(base, d, 0.0)[15, 16])(0.2)
    assert np.isfinite(g)


# --- point-source placement ---------------------------------------------------

from forwardgal.psf import make_point_source_renderer  # noqa: E402


def asym_kernel(n=21):
    k = gaussian_psf(n, sigma=1.8, dy=0.3, dx=-0.4)  # asymmetric: catches flips
    return k / k.sum()


def test_whole_pixel_placement_reproduces_stamp():
    k = asym_kernel()
    img = np.asarray(make_point_source_renderer(k, (40, 50))(17.0, 23.0))
    np.testing.assert_allclose(img[17 - 10 : 17 + 11, 23 - 10 : 23 + 11], k, atol=1e-14)
    assert img.sum() == pytest.approx(1.0, rel=1e-12)


def test_fractional_placement_equals_fourier_shift():
    k = gaussian_psf(41, sigma=2.5)
    k /= k.sum()
    img = np.asarray(make_point_source_renderer(k, (41, 41))(20.3, 19.6))
    np.testing.assert_allclose(img, np.asarray(fourier_shift(k, 0.3, -0.4)), atol=1e-12)


def test_psf_reference_point_lands_on_target():
    # The PSF was "made for" a source at stamp position (10.25, 9.6): asking for
    # a point source at grid (17.25, 22.6) must need no fractional shift.
    k = asym_kernel()
    img = np.asarray(make_point_source_renderer(k, (40, 50), psf_ref=(10.25, 9.6))(17.25, 22.6))
    np.testing.assert_allclose(img[7:28, 13:34], k, atol=1e-13)


def test_point_source_differentiable_in_position():
    k = gaussian_psf(21, sigma=2.5)
    k /= k.sum()
    render = make_point_source_renderer(k, (30, 30))
    g = jax.grad(lambda r: render(r, 15.0)[16, 15])(15.2)
    assert np.isfinite(g) and g > 0  # moving the source towards row 16 brightens it


def test_fractional_placement_independent_of_box_parity():
    # With odd FFT lengths the shift is a pure (trigonometric-interpolation)
    # shift, whatever the parity of the box: even and odd boxes agree.
    k = np.zeros((21, 21))
    k[10, 10], k[10, 9], k[10, 11] = 0.6, 0.2, 0.2  # sharp core
    a = np.asarray(make_point_source_renderer(k, (40, 50))(17.0, 23.03))
    b = np.asarray(make_point_source_renderer(k, (41, 51))(17.0, 23.03))
    np.testing.assert_allclose(a[5:30, 5:40], b[5:30, 5:40], atol=1e-6)


def test_sinc_tails_of_a_sharp_kernel_scale_with_the_shift():
    # Documented behaviour: a fractional shift of a non-band-limited kernel
    # leaves alternating tails along its row, roughly proportional to the shift;
    # a whole-pixel placement leaves none.
    k = np.zeros((21, 21))
    k[10, 10], k[10, 9], k[10, 11] = 0.6, 0.2, 0.2
    render = make_point_source_renderer(k, (41, 81))
    tail = lambda d: np.abs(np.asarray(render(20.0, 30.0 + d))[20, 55:]).max()  # noqa: E731
    assert tail(0.0) < 1e-12
    assert 1.5 < tail(0.04) / tail(0.02) < 2.5
