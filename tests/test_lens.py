import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from forwardgal.lens import bilinear, load_deflection

RA0, DEC0 = 178.330251, 7.932518
STEP_DEG = 3.49333333333333e-05  # 0.12576"/px, as in the Lenstool maps


def write_map(path, data, cdelt1=-STEP_DEG, pc12=0.0):
    """Write a 101x101 map with a Lenstool-like header (reference at the centre)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [51.0, 51.0]  # 1-based -> 0-based [50, 50]
    w.wcs.cdelt = [cdelt1, STEP_DEG]
    w.wcs.pc = [[1.0, pc12], [0.0, 1.0]]
    fits.PrimaryHDU(data.astype(np.float32), header=w.to_header()).writeto(path)


@pytest.fixture
def field(tmp_path):
    # Fill the maps with recognisable values: alpha_x = column, alpha_y = row.
    rows, cols = np.mgrid[0:101, 0:101]
    write_map(tmp_path / "x.fits", cols)
    write_map(tmp_path / "y.fits", rows)
    return load_deflection(tmp_path / "x.fits", tmp_path / "y.fits", z_ref=1.524)


def test_reference_pixel_is_origin(field):
    assert (field.i0, field.j0) == (50.0, 50.0)
    tx, ty = field.pixel_to_theta(50, 50)
    assert (tx, ty) == (0.0, 0.0)


def test_axes_point_east_and_north(field):
    # One column to the right (west) -> theta_x decreases by one step.
    tx, ty = field.pixel_to_theta(50, 51)
    assert tx == pytest.approx(-STEP_DEG * 3600) and ty == 0.0
    # One row up (north) -> theta_y increases by one step.
    tx, ty = field.pixel_to_theta(51, 50)
    assert tx == 0.0 and ty == pytest.approx(STEP_DEG * 3600)


def test_maps_indexed_row_col(field):
    # alpha_x was filled with the column index, alpha_y with the row index.
    # By default the x map is treated as pointing west, so it is negated.
    assert field.alpha_x[10, 30] == -30.0
    assert field.alpha_y[10, 30] == 10.0


def test_x_points_west_false_keeps_sign(tmp_path):
    rows, cols = np.mgrid[0:101, 0:101]
    write_map(tmp_path / "x.fits", cols)
    write_map(tmp_path / "y.fits", rows)
    f = load_deflection(tmp_path / "x.fits", tmp_path / "y.fits", 1.5, x_points_west=False)
    assert f.alpha_x[10, 30] == 30.0


def test_pixel_theta_roundtrip(field):
    row, col = np.array([0.0, 12.3, 99.9]), np.array([100.0, 47.5, 3.2])
    r2, c2 = field.theta_to_pixel(*field.pixel_to_theta(row, col))
    np.testing.assert_allclose(r2, row, atol=1e-12)
    np.testing.assert_allclose(c2, col, atol=1e-12)


def test_sky_to_theta_small_offsets(field):
    # 10" east and 5" north of the reference: tangent-plane theta should match
    # the small-angle formula (dRA cos Dec0, dDec) to well below 1 mas.
    ra = RA0 + 10.0 / 3600 / np.cos(np.radians(DEC0))
    dec = DEC0 + 5.0 / 3600
    tx, ty = field.sky_to_theta(ra, dec)
    assert tx == pytest.approx(10.0, abs=1e-3)
    assert ty == pytest.approx(5.0, abs=1e-3)


def test_sky_theta_roundtrip(field):
    ra, dec = field.theta_to_sky(np.array([17.8, -4.6]), np.array([3.6, 16.1]))
    tx, ty = field.sky_to_theta(ra, dec)
    np.testing.assert_allclose(tx, [17.8, -4.6], atol=1e-9)
    np.testing.assert_allclose(ty, [3.6, 16.1], atol=1e-9)


def test_rotated_wcs_rejected(tmp_path):
    data = np.zeros((101, 101))
    write_map(tmp_path / "x.fits", data, pc12=0.1)
    write_map(tmp_path / "y.fits", data, pc12=0.1)
    with pytest.raises(ValueError, match="rotated"):
        load_deflection(tmp_path / "x.fits", tmp_path / "y.fits", z_ref=1.5)


# --- M2b: bilinear interpolation ---------------------------------------------

def test_bilinear_hits_grid_points():
    arr = np.arange(20.0).reshape(4, 5)
    rows, cols = np.mgrid[0:4, 0:5]
    np.testing.assert_array_equal(bilinear(arr, rows, cols), arr)


def test_bilinear_exact_for_bilinear_function():
    # f = a + b*row + c*col + d*row*col is reproduced exactly, including edges.
    rows, cols = np.mgrid[0:6, 0:7].astype(float)
    f = lambda r, c: 1.5 - 2.0 * r + 0.7 * c + 0.3 * r * c  # noqa: E731
    arr = f(rows, cols)
    r = np.array([0.0, 0.25, 2.5, 4.99, 5.0, 3.3])
    c = np.array([0.0, 5.75, 3.1, 0.01, 6.0, 6.0])
    np.testing.assert_allclose(bilinear(arr, r, c), f(r, c), rtol=0, atol=1e-12)


def test_bilinear_outside_is_nan():
    arr = np.ones((4, 5))
    out = bilinear(arr, np.array([-0.01, 3.01, 1.0, 1.0]), np.array([1.0, 1.0, -0.5, 4.5]))
    assert np.all(np.isnan(out))


def test_alpha_and_backtrace(field):
    # In the fixture, the stored alpha_x = -col and alpha_y = row (both linear),
    # so interpolation is exact at any theta inside the map.
    tx, ty = np.array([1.0, -3.3]), np.array([2.2, 0.4])
    row, col = field.theta_to_pixel(tx, ty)
    ax, ay = field.alpha(tx, ty)
    np.testing.assert_allclose(ax, -col, atol=1e-10)
    np.testing.assert_allclose(ay, row, atol=1e-10)
    bx, by = field.backtrace(tx, ty)
    np.testing.assert_allclose(bx, tx - ax, atol=1e-12)
    np.testing.assert_allclose(by, ty - ay, atol=1e-12)
