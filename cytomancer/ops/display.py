import numpy as np
import xarray as xr
from PIL import ImageColor
from skimage import color, exposure, util  # type: ignore


def rescale_intensity(arr: xr.DataArray, dims: list[str], **kwargs):
    def _rescale_intensity(
        frame, in_percentile: tuple[int, int] | None = None, **kwargs
    ):
        if in_percentile is not None:
            l, h = np.percentile(frame, in_percentile)  # noqa: E741
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
        dask_gufunc_kwargs={"allow_rechunk": True},
        vectorize=True,
        dask="parallelized",
    )


def clahe(arr: xr.DataArray, clip_limit: float):
    def _clahe(frame, clip_limit):
        rescaled = exposure.rescale_intensity(frame, out_range=(0, 1))
        return exposure.equalize_adapthist(rescaled, clip_limit=clip_limit)

    return xr.apply_ufunc(
        _clahe,
        arr,
        kwargs={"clip_limit": clip_limit},
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
    )


def apply_psuedocolor(arr: xr.DataArray) -> xr.DataArray:
    def _get_float_color(hexcode: str):
        rgb = tuple(map(float, ImageColor.getcolor(hexcode, "RGB")))  # type: ignore[arg-type]
        max_val = max(rgb)
        rgb_corrected = tuple(x / max_val for x in rgb)
        return rgb_corrected

    if "metadata" in arr.attrs:
        channels = arr.attrs["metadata"]["metadata"].channels
        color_codes = {}
        for channel in channels:
            intcode = channel.channel.colorRGB
            rgb = (intcode & 255, (intcode >> 8) & 255, (intcode >> 16) & 255)
            max_val = max(rgb)
            float_color = tuple(x / max_val for x in rgb)
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
        dask_gufunc_kwargs={"output_sizes": {"rgb": 3}},
        output_dtypes=[np.uint8],
        vectorize=True,
        dask="parallelized",
    )

    return rgb.transpose(..., "rgb")
