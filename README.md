# dask-rasterio

[![Build Status](https://travis-ci.org/dymaxionlabs/dask-rasterio.svg?branch=master)](https://travis-ci.org/dymaxionlabs/dask-rasterio)
[![codecov](https://codecov.io/gh/dymaxionlabs/dask-rasterio/branch/master/graph/badge.svg)](https://codecov.io/gh/dymaxionlabs/dask-rasterio) [![Join the chat at https://gitter.im/dymaxionlabs/dask-rasterio](https://badges.gitter.im/dymaxionlabs/dask-rasterio.svg)](https://gitter.im/dymaxionlabs/dask-rasterio?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

`dask-rasterio` provides some methods for reading and writing rasters in
parallel using [Rasterio](https://github.com/mapbox/rasterio) and
[Dask](https://dask.pydata.org) arrays.


## Usage

#### Read a multiband raster

```python
>>> from dask_rasterio import read_raster

>>> array = read_raster('tests/data/RGB.byte.tif')
>>> array
dask.array<stack, shape=(3, 718, 791), dtype=uint8, chunksize=(1, 3, 791)>

>>> array.mean()
dask.array<mean_agg-aggregate, shape=(), dtype=float64, chunksize=()>
>>> array.mean().compute()
40.858976977533935
```

#### Read a single band from a raster

```python
>>> from dask_rasterio import read_raster

>>> array = read_raster('tests/data/RGB.byte.tif', band=3)
>>> array
dask.array<raster, shape=(718, 791), dtype=uint8, chunksize=(3, 791)>
```

#### Write a singleband or multiband raster

```python
>>> from dask_rasterio import read_raster, write_raster

>>> array = read_raster('tests/data/RGB.byte.tif')

>>> new_array = array & (array > 100)
>>> new_array
dask.array<and_, shape=(3, 718, 791), dtype=uint8, chunksize=(1, 3, 791)>

>>> prof = ... # reuse profile from tests/data/RGB.byte.tif...
>>> write_raster('processed_image.tif', new_array, **prof)
```

#### Chunk size

Both `read_raster` and `write_raster` accept a `block_size` argument that
acts as a multiplier to the block size of rasters. The default value is 1,
which means the dask array chunk size will be the same as the block size of
the raster file. You will have to adjust this value depending on the
specification of your machine (how much memory do you have, and the block
size of the raster).


## Install

Install with pip:

```
pip install dask-rasterio
```

## Development

This project is managed by [Poetry](https://github.com/sdispater/poetry).  If
you do not have it installed, please refer to 
[Poetry instructions](https://github.com/sdispater/poetry#installation).

Now, clone the repository and run `poetry install`.  This will create a virtual
environment and install all required packages there.

Run `poetry run pytest` to run all tests.

Run `poetry build` to build package on `dist/`.


## Issue tracker

Please report any bugs and enhancement ideas using the GitHub issue tracker:

  https://github.com/dymaxionlabs/dask-rasterio/issues

Feel free to also ask questions on our
[Gitter channel](https://gitter.im/dymaxionlabs/dask-rasterio), or by email.


## Help wanted

Any help in testing, development, documentation and other tasks is highly
appreciated and useful to the project.

For more details, see the file [CONTRIBUTING.md](CONTRIBUTING.md).


## License

Source code is released under a BSD-2 license.  Please refer to
[LICENSE.md](LICENSE.md) for more information.
