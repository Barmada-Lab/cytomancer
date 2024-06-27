import pathlib as pl

from skimage import transform
import xarray as xr
import nd2

from cytomancer.experiment import Axes


def load_nd2(path: pl.Path) -> xr.DataArray:

    arr = nd2.imread(path, xarray=True)
    nd2_label = path.name.replace(".nd2", "")
    arr = arr.expand_dims(Axes.REGION).assign_coords({Axes.REGION: [nd2_label]})

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
        C=Axes.CHANNEL,
        P=Axes.FIELD,
        Y=Axes.Y,
        X=Axes.X,)

    if "T" in arr.dims:
        rename_dict["T"] = Axes.TIME
    if "Z" in arr.dims:
        rename_dict["Z"] = Axes.Z
    arr = arr.rename(rename_dict)

    return arr


def load_nd2_collection(base_path: pl.Path) -> xr.DataArray:

    paths = list(base_path.glob("*.nd2"))
    regions = [path.name.replace(".nd2", "") for path in paths]

    arrs = []
    for path in paths:

        nd2 = load_nd2(path)
        arrs.append(nd2)

    assert len(set(arr.sizes[Axes.CHANNEL] for arr in arrs)) == 1, "Number of channels must be the same across all images"

    aspect_ratios = [nd2.sizes[Axes.Y] / nd2.sizes[Axes.X] for nd2 in arrs]
    assert len(set(aspect_ratios)) == 1, "Aspect ratios must be the same across all images"

    max_x = max(nd2.sizes[Axes.X] for nd2 in arrs)
    max_y = max(nd2.sizes[Axes.Y] for nd2 in arrs)

    homogenized = []
    for arr in arrs:

        if nd2.sizes[Axes.Y] == max_y and nd2.sizes[Axes.X] == max_x:
            homogenized.append(arr)
            continue

        resized = xr.apply_ufunc(
            transform.resize,
            arr,
            input_core_dims=[[Axes.Y, Axes.X]],
            output_core_dims=[[Axes.Y, Axes.X]],
            dask="parallelized",
            vectorize=True,
            dask_gufunc_kwargs={"output_sizes": {Axes.Y: max_y, Axes.X: max_x}},
            kwargs=dict(output_shape=(max_y, max_x)))

        homogenized.append(resized)

    return xr.concat(homogenized, dim=Axes.REGION).assign_coords({Axes.REGION: regions})
