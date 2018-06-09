import dask.array as da
import rasterio
from dask import is_dask_collection
from dask.base import tokenize
from rasterio.windows import Window


def read_raster(path, band=None, block_size=1):
    """Read all or some bands from raster

    Arguments:
        path {string} -- path to raster file

    Keyword Arguments:
        band {int, iterable(int)} -- band number or iterable of bands.
            When passing None, it reads all bands (default: {None})
        block_size {int} -- block size multiplier (default: {1})

    Returns:
        dask.array.Array -- a Dask array
    """

    if isinstance(band, int):
        return read_raster_band(path, band=band, block_size=block_size)
    else:
        if band is None:
            bands = range(1, get_band_count(path) + 1)
        else:
            bands = list(band)
        return da.stack([
            read_raster_band(path, band=band, block_size=block_size)
            for band in bands
        ])


def read_raster_band(path, band=1, block_size=1):
    """Read a raster band and return a Dask array

    Arguments:
        path {string} -- path to the raster file

    Keyword Arguments:
        band {int} -- number of band to read (default: {1})
        block_size {int} -- block size multiplier (default: {1})

    """

    def read_window(raster_path, window, band):
        with rasterio.open(raster_path) as src:
            return src.read(band, window=window)

    def resize_window(window, block_size):
        return Window(
            col_off=window.col_off * block_size,
            row_off=window.row_off * block_size,
            width=window.width * block_size,
            height=window.height * block_size)

    def block_windows(dataset, band, block_size):
        return [(pos, resize_window(win, block_size))
                for pos, win in dataset.block_windows(band)]

    with rasterio.open(path) as src:
        h, w = src.block_shapes[band - 1]
        chunks = (h * block_size, w * block_size)
        name = 'raster-{}'.format(tokenize(path, band, chunks))
        dtype = src.dtypes[band - 1]
        shape = src.shape
        blocks = block_windows(src, band, block_size)

    dsk = {(name, i, j): (read_window, path, window, band)
           for (i, j), window in blocks}

    return da.Array(dsk, name, chunks, dtype, shape)


def get_band_count(raster_path):
    """Read raster band count"""
    with rasterio.open(raster_path) as src:
        return src.count
