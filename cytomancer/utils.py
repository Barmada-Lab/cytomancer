import random
from collections.abc import Generator
from itertools import product
from typing import TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)


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
