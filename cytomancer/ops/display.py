from PIL import ImageColor
from skimage import exposure, color, util, morphology, filters  # type: ignore
import pandas as pd
import xarray as xr
import numpy as np


def rescale_intensity(arr: xr.DataArray, dims: list[str], **kwargs):

    def _rescale_intensity(frame, in_percentile: tuple[int, int] | None = None, **kwargs):
        if in_percentile is not None:
            l, h = np.percentile(frame, in_percentile)  # type: ignore
            kwargs.pop("in_range", None)
            return exposure.rescale_intensity(frame, in_range=(l, h), **kwargs)
        else:
            return exposure.rescale_intensity(frame, **kwargs)

    return xr.apply_ufunc(
        _rescale_intensity,
        arr,
        kwargs=kwargs,
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        vectorize=True,
        dask="parallelized")


def clahe(arr: xr.DataArray):

    def _clahe(frame):
        rescaled = exposure.rescale_intensity(frame, out_range=np.uint8)
        footprint = morphology.disk(3)
        blurred = filters.median(rescaled, footprint)
        clahe = exposure.equalize_adapthist(blurred, 128, clip_limit=0.03)
        return clahe

    return xr.apply_ufunc(
        _clahe,
        arr,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized")


def _get_float_color(hexcode: str):
    rgb = tuple(map(float, ImageColor.getcolor(hexcode, "RGB")))
    max_val = max(rgb)
    rgb_corrected = tuple(map(lambda x: x / max_val, rgb))
    return rgb_corrected


def apply_psuedocolor(arr: xr.DataArray):

    if "metadata" in arr.attrs:
        channels = arr.attrs["metadata"]["metadata"].channels
        color_codes = {}
        for channel in channels:
            intcode = channel.channel.colorRGB
            rgb = (intcode & 255, (intcode >> 8) & 255, (intcode >> 16) & 255)
            max_val = max(rgb)
            float_color = tuple(map(lambda x: x / max_val, rgb))
            color_codes[channel.channel.name] = float_color
    else:
        color_codes = {
            "DAPI": _get_float_color("#007fff"),
            "RFP": _get_float_color("#ffe600"),
            "GFP": _get_float_color("#00ff00"),
            "Cy5": _get_float_color("#ff0000"),
            "white_light": _get_float_color("#ffffff"),
        }

    def _rgb(frame, channel):
        color_code = color_codes.get(str(channel), (1.0, 1.0, 1.0))
        float_frame = util.img_as_float(frame)
        rgb_frame = color.gray2rgb(float_frame)
        colored = rgb_frame * color_code
        return exposure.rescale_intensity(colored, out_range="uint8")

    rgb = xr.apply_ufunc(
        _rgb,
        arr,
        arr["channel"],
        input_core_dims=[["y", "x"], []],
        output_core_dims=[["y", "x", "rgb"]],
        dask_gufunc_kwargs=dict(output_sizes={"rgb": 3}),
        output_dtypes=[np.uint8],
        vectorize=True,
        dask="parallelized")

    return rgb.transpose(..., "rgb")


def stitch(arr: xr.DataArray, trim: float = 0.05):
    # TODO: arrange tiles correctly
    trimmed = arr.sel(
        y=slice(int(arr["y"].size * trim), int(arr["y"].size * (1 - trim))),
        x=slice(int(arr["x"].size * trim), int(arr["x"].size * (1 - trim)))
    )
    field_dim = np.sqrt(arr["field"].size).astype(int)
    mi = pd.MultiIndex.from_product(
        (range(field_dim), range(field_dim)), names=["fx", "fy"])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mi, "field")
    trimmed = trimmed.assign_coords(mindex_coords).unstack("field")

    x_stitched = xr.concat(trimmed.transpose("fx", ..., "y", "x")[::-1], dim="y")
    stitched = xr.concat(x_stitched.transpose("fy", ..., "x", "y")[::-1], dim="x")
    return stitched.drop(["fx", "fy"])
