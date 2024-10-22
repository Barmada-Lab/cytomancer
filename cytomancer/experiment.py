from enum import StrEnum, auto

from PIL import ImageColor

from cytomancer.enumero import NaturalOrderStrEnum


class Axes(StrEnum):
    region = auto()
    field = auto()
    channel = auto()
    time = auto()
    y = auto()
    x = auto()
    z = auto()


class ExperimentType(NaturalOrderStrEnum):
    CQ1 = "cq1"
    ND2 = "nd2"
    LUX = "lux"
    LEGACY = "legacy"
    LEGACY_ICC = "legacy-icc"


def _get_float_color(hexcode: str):
    rgb = tuple(map(float, ImageColor.getcolor(hexcode, "RGB")))
    max_val = max(rgb)
    rgb_corrected = tuple(x / max_val for x in rgb)
    return rgb_corrected


_float_colors = {
    "DAPI": "#007fff",
    "RFP": "#ffe600",
    "GFP": "#00ff00",
    "Cy5": "#ff0000",
    "white_light": "#ffffff",
}


def get_float_color(channel: str):
    if channel in _float_colors:
        return _get_float_color(_float_colors[channel])
    else:
        raise ValueError(f"Channel {channel} is not known")
