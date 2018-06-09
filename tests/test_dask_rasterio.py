import os
import tempfile

import dask.array as da
import numpy as np
import pytest
import rasterio
from numpy.testing import assert_array_equal

from dask_rasterio import (__version__, read_raster, read_raster_band,
                           write_raster)
from dask_rasterio.read import get_band_count

THRESHOLD = 127


def get_profile(path):
    with rasterio.open(path) as src:
        return src.profile.copy()


def assert_equal_raster_profile(dataset, expected_profile):
    attributes = ['transform', 'crs', 'driver', 'width', 'height', 'count']
    for attr in attributes:
        assert getattr(dataset, attr) == expected_profile[attr]


@pytest.fixture
def some_raster_path():
    return os.path.join(os.path.dirname(__file__), 'data', 'RGB.byte.tif')


def test_version():
    assert __version__ == '0.1.0'


def test_read_raster(some_raster_path):
    array = read_raster(some_raster_path)
    assert isinstance(array, da.Array)

    with rasterio.open(some_raster_path) as src:
        expected_array = src.read()
        assert array.shape == expected_array.shape
        assert array.dtype == expected_array.dtype
        assert_array_equal(array.compute(), expected_array)


def test_read_raster_band(some_raster_path):
    g_array = read_raster_band(some_raster_path, 2)
    assert isinstance(g_array, da.Array)

    with rasterio.open(some_raster_path) as src:
        expected_array = src.read(2)
        assert g_array.shape == expected_array.shape
        assert g_array.dtype == expected_array.dtype
        assert_array_equal(g_array.compute(), expected_array)


def test_read_raster_single_band(some_raster_path):
    array = read_raster(some_raster_path, band=3)
    assert isinstance(array, da.Array)

    expected_array = read_raster_band(some_raster_path, band=3)
    assert array.shape == expected_array.shape
    assert array.dtype == expected_array.dtype
    assert_array_equal(array.compute(), expected_array.compute())


def test_read_raster_multi_band(some_raster_path):
    array = read_raster(some_raster_path, band=(1, 3))
    assert isinstance(array, da.Array)

    expected_array = da.stack([
        read_raster_band(some_raster_path, band=1),
        read_raster_band(some_raster_path, band=3)
    ])
    assert array.shape == expected_array.shape
    assert array.dtype == expected_array.dtype
    assert_array_equal(array.compute(), expected_array.compute())


def test_do_calcs_on_array(some_raster_path):
    r_array = read_raster_band(some_raster_path, 1)
    mean = np.mean(r_array)
    assert isinstance(mean, da.Array)

    with rasterio.open(some_raster_path) as src:
        expected_array = src.read(1)
        expected_mean = np.mean(expected_array)
        assert mean.compute() == expected_mean


def test_write_raster_band(some_raster_path):
    with tempfile.TemporaryDirectory(prefix='dask_rasterio_test_') as tmpdir:
        # Read first band of raster
        array = read_raster_band(some_raster_path, 1)

        # Generate new data
        new_array = array & (array > THRESHOLD)

        # Build a profile for the new single-band GeoTIFF
        prof = get_profile(some_raster_path)
        prof.update(count=1)

        # Write raster file
        dst_path = os.path.join(tmpdir, 'test.tif')
        write_raster(dst_path, new_array, **prof)

        with rasterio.open(dst_path) as src:
            assert_equal_raster_profile(src, prof)
            expected_new_array = src.read(1)
            assert expected_new_array.dtype == new_array.dtype
            assert_array_equal(new_array.compute(), expected_new_array)


def test_write_raster(some_raster_path):
    with tempfile.TemporaryDirectory(prefix='dask_rasterio_test_') as tmpdir:
        array = read_raster(some_raster_path)
        new_array = array & (array > THRESHOLD)

        prof = get_profile(some_raster_path)

        dst_path = os.path.join(tmpdir, 'test.tif')
        write_raster(dst_path, new_array, **prof)

        with rasterio.open(dst_path) as src:
            assert_equal_raster_profile(src, prof)
            expected_new_array = src.read()
            assert expected_new_array.dtype == new_array.dtype
            assert_array_equal(new_array.compute(), expected_new_array)


def test_write_raster_band_from_numpy(some_raster_path):
    with tempfile.TemporaryDirectory(prefix='dask_rasterio_test_') as tmpdir:
        # Read first band of raster with Rasterio
        with rasterio.open(some_raster_path) as src:
            array = src.read(1)

        # Generate new data
        new_array = array & (array > THRESHOLD)

        # Build a profile for the new single-band GeoTIFF
        prof = get_profile(some_raster_path)
        prof.update(count=1)

        # Write raster file
        dst_path = os.path.join(tmpdir, 'test.tif')
        write_raster(dst_path, new_array, **prof)

        with rasterio.open(dst_path) as src:
            assert_equal_raster_profile(src, prof)
            expected_new_array = src.read(1)
            assert expected_new_array.dtype == new_array.dtype
            assert_array_equal(new_array, expected_new_array)


def test_write_raster_from_numpy(some_raster_path):
    with tempfile.TemporaryDirectory(prefix='dask_rasterio_test_') as tmpdir:
        with rasterio.open(some_raster_path) as src:
            array = src.read()

        new_array = array & (array > THRESHOLD)

        prof = get_profile(some_raster_path)

        dst_path = os.path.join(tmpdir, 'test.tif')
        write_raster(dst_path, new_array, **prof)

        with rasterio.open(dst_path) as src:
            assert_equal_raster_profile(src, prof)
            expected_new_array = src.read()
            assert expected_new_array.dtype == new_array.dtype
            assert_array_equal(new_array, expected_new_array)


def test_cannot_write_raster_with_badly_shaped_array(some_raster_path):
    with tempfile.TemporaryDirectory(prefix='dask_rasterio_test_') as tmpdir:
        prof = get_profile(some_raster_path)
        dst_path = os.path.join(tmpdir, 'test.tif')

        with pytest.raises(TypeError):
            write_raster(dst_path, np.random.rand(10), **prof)

        with pytest.raises(TypeError):
            write_raster(dst_path, np.random.rand(10, 10, 10, 3), **prof)


def multiply_chunks(chunks, multiplier):
    w_chunks = tuple(np.array([list(chunks[0])])) * multiplier
    h_chunks = tuple(np.array([list(chunks[1])])) * multiplier
    return (w_chunks, h_chunks)


def test_read_raster_band_with_block_size(some_raster_path):
    array = read_raster_band(some_raster_path)
    array_4b = read_raster_band(some_raster_path, block_size=4)
    assert array.shape == array_4b.shape
    assert array.dtype == array_4b.dtype
    assert_array_equal(array, array_4b)

    with rasterio.open(some_raster_path) as src:
        ch, cw = src.block_shapes[0]

    assert array.chunks[0][0] == ch
    assert array.chunks[1][0] == cw
    assert array_4b.chunks[0][0] == ch * 4
    assert array_4b.chunks[1][0] == cw * 4