import logging
import pathlib as pl

import dask
import dask.array as da
import numpy as np
import tifffile
from skimage import exposure, transform  # type: ignore

logger = logging.getLogger(__name__)


def read_tiff_delayed(shape: tuple, reshape: bool = True):
    def read(path: pl.Path) -> np.ndarray:
        try:
            logger.debug(f"Reading {path}")
            img = tifffile.imread(path)
            if img.shape != shape and reshape:
                img = transform.resize(
                    img, shape, preserve_range=True, anti_aliasing=True
                )
            elif img.shape != shape and not reshape:
                raise ValueError(
                    f"Image shape {img.shape} does not match expected shape {shape}; you can pass reshape=True to resize the image to a standard shape."
                )
            return exposure.rescale_intensity(img, out_range=np.float32)
        except (ValueError, NameError, FileNotFoundError) as e:
            logger.warning(
                f"Error reading {path}: {e}\nThis field will be filled based on surrounding fields and timepoints."
            )
            img = np.zeros(shape, dtype=np.float32)
            img[:] = np.nan
            return img

    return dask.delayed(read)


def read_tiff_toarray(path: pl.Path, shape: tuple = (1024, 1024)):
    return da.from_delayed(read_tiff_delayed(shape)(path), shape, dtype=np.float32)
