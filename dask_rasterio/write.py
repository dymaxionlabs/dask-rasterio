import dask.array as da
import rasterio
from dask import is_dask_collection
from dask.base import tokenize
from rasterio.windows import Window


def write_raster(path, array, **kwargs):
    """Write a dask array to a raster file

    If array is 2d, write array on band 1.
    If array is 3d, write data on each band

    Arguments:
        path {string} -- path of raster to write
        array {dask.array.Array} -- band array
        kwargs {dict} -- keyword arguments to delegate to rasterio.open

    Examples:
        # Write a single band raster
        >> red_band = read_raster_band("test.tif", band=1)
        >> write_raster("new.tif", red_band)

        # Write a multiband raster
        >> img = read_raster("test.tif")
        >> new_img = process(img)
        >> write_raster("new.tif", new_img)

    """
    if len(array.shape) != 2 and len(array.shape) != 3:
        raise TypeError('invalid shape (must be either 2d or 3d)')

    if is_dask_collection(array):
        with RasterioDataset(path, 'w', **kwargs) as dst:
            da.store(array, dst, lock=True)
    else:
        with rasterio.open(path, 'w', **kwargs) as dst:
            if len(array.shape) == 2:
                dst.write(array, 1)
            else:
                dst.write(array)


class RasterioDataset:
    """Rasterio wrapper to allow dask.array.store to do window saving.

    Example:
        >> rows = cols = 21696
        >> a = da.ones((4, rows, cols), dtype=np.float64, chunks=(1, 4096, 4096) )
        >> a = a * np.array([255., 255., 255., 255.])[:, None, None]
        >> a = a.astype(np.uint8)
        >> with RasterioDataset('test.tif', 'w', driver='GTiff', width=cols, height=rows, count=4, dtype=np.uint8) as r_file:
        ..    da.store(a, r_file, lock=True)
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dataset = None

    def __setitem__(self, key, item):
        """Put the data chunk in the image"""
        if len(key) == 3:
            index_range, y, x = key
            indexes = list(
                range(index_range.start + 1, index_range.stop + 1,
                      index_range.step or 1))
        else:
            indexes = 1
            y, x = key

        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start

        self.dataset.write(
            item, window=Window(chx_off, chy_off, chx, chy), indexes=indexes)

    def __enter__(self):
        """Enter method"""
        self.dataset = rasterio.open(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method"""
        self.dataset.close()
