import pathlib as pl
import random
from collections.abc import Generator
from itertools import product
from typing import TypeVar

import xarray as xr

from cytomancer.experiment import ExperimentType
from cytomancer.io.cq1_loader import load_cq1
from cytomancer.io.legacy_loader import load_legacy, load_legacy_icc
from cytomancer.io.lux_loader import load_lux
from cytomancer.io.nd2_loader import load_nd2_collection

T = TypeVar("T", xr.DataArray, xr.Dataset)


def load_experiment(
    path: pl.Path, experiment_type: ExperimentType, fillna: bool = False
) -> xr.DataArray:
    match experiment_type:
        case ExperimentType.LEGACY:
            return load_legacy(path, fillna)
        case ExperimentType.LEGACY_ICC:
            return load_legacy_icc(path, fillna)
        case ExperimentType.ND2:
            return load_nd2_collection(path)
        case ExperimentType.LUX:
            return load_lux(path, fillna)
        case ExperimentType.CQ1:
            return load_cq1(path)


def apply_ufunc_xy(func, arr: xr.DataArray, ufunc_kwargs=None, **kwargs):
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    return xr.apply_ufunc(
        func,
        arr,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
        kwargs=ufunc_kwargs,
        **kwargs,
    )


def iter_idx_prod(arr: T, subarr_dims=None, shuffle=False) -> Generator[T, None, None]:
    """
    Iterates over the product of an array's indices. Can be used to iterate over
    all the (coordinate-less) XY(Z) planes in an experiment.
    """
    if subarr_dims is None:
        subarr_dims = []
    indices = [name for name in arr.indexes if name not in subarr_dims]
    idxs = list(product(*[arr.indexes[name] for name in indices]))
    if shuffle:
        random.shuffle(idxs)
    for coords in idxs:
        selector = dict(zip(indices, coords, strict=False))
        yield arr.sel(selector)


def get_user_confirmation(prompt, default=None):
    """
    Prompt the user for a yes/yo response.

    Args:
        prompt (str): The question to ask the user.
        default (str, optional): The default response if the user provides no input.
                                 Should be 'y'/'yes', 'n'/'no', or None. Defaults to None.

    Returns:
        bool: True if the user confirms (Yes), False otherwise (No).
    """

    # Establish valid responses
    yes_responses = {"yes", "y"}
    no_responses = {"no", "n"}

    # Include default in the prompt if it is provided
    if default is not None:
        default = default.lower()
        if default in yes_responses:
            prompt = f"{prompt} [Y/n]: "
        elif default in no_responses:
            prompt = f"{prompt} [y/N]: "
    else:
        prompt = f"{prompt} [y/n]: "

    while True:
        response = input(prompt).strip().lower()

        # Check for a valid response; if found, return True/False
        if response in yes_responses:
            return True
        if response in no_responses:
            return False
        if default is not None and response == "":
            return default in yes_responses

        # If response is invalid, notify the user and prompt again
        print("Please respond with 'y' or 'n' (or 'yes' or 'no').")
