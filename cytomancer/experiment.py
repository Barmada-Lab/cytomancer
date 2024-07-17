from cytomancer.enumero import NaturalOrderStrEnum
from enum import auto, StrEnum


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
