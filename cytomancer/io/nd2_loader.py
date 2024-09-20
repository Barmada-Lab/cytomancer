import pathlib as pl

from skimage import transform
import xarray as xr
import nd2


def load_nd2(path: pl.Path) -> xr.DataArray:

    arr = nd2.imread(path, xarray=True, dask=True)
    nd2_label = path.name.replace(".nd2", "")
    arr = arr.expand_dims("region").assign_coords({"region": [nd2_label]})

    # single-channel images don't include C
    if "C" not in arr.dims:
        arr = arr.expand_dims("C")
        channel = arr.metadata["metadata"].channels[0].channel.name.strip()
        arr = arr.assign_coords(C=[channel])
    else:
        # sanitize inputs that may contain leading/trailing spaces
        arr = arr.assign_coords(C=[channel.strip() for channel in arr.C.values])

    # single-field images don't include P
    if "P" not in arr.dims:
        arr = arr.expand_dims("P")
        point_coords = ["0"]
    else:
        point_coords = list(map(str, range(arr.P.size)))

    arr = arr.assign_coords(P=point_coords)
    rename_dict = dict(
        C="channel",
        P="field",
        Y="y",
        X="x",)

    if "T" in arr.dims:
        rename_dict["T"] = "time"
    if "Z" in arr.dims:
        rename_dict["Z"] = "z"
    arr = arr.rename(rename_dict)
    arr.attrs = {}
    return arr


def load_nd2_collection(base_path: pl.Path) -> xr.DataArray:

    paths = list(base_path.glob("*.nd2"))
    regions = [path.name.replace(".nd2", "") for path in paths]

    arrs = []
    for path in paths:
        nd2 = load_nd2(path)
        arrs.append(nd2)

    assert len(set(arr.sizes["channel"] for arr in arrs)) == 1, "Number of channels must be the same across all images"

    aspect_ratios = [nd2.sizes["y"] / nd2.sizes["x"] for nd2 in arrs]
    assert len(set(aspect_ratios)) == 1, "Aspect ratios must be the same across all images"

    max_x = max(nd2.sizes["x"] for nd2 in arrs)
    max_y = max(nd2.sizes["y"] for nd2 in arrs)

    homogenized = []
    for arr in arrs:

        if nd2.sizes["y"] == max_y and nd2.sizes["x"] == max_x:
            homogenized.append(arr)
            continue

        resized = xr.apply_ufunc(
            transform.resize,
            arr,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            dask="parallelized",
            vectorize=True,
            dask_gufunc_kwargs={"output_sizes": {"y": max_y, "x": max_x}},
            kwargs=dict(output_shape=(max_y, max_x)))

        homogenized.append(resized)

    return xr.concat(homogenized, dim="region").assign_coords({"region": regions})
