from pathlib import Path

import joblib
import xarray as xr
from sklearn.pipeline import Pipeline

from cytomancer.oneoffs import ilastish_seg_model


def neurite_seg(gfp: xr.DataArray, model: Pipeline):

    def segment(field):
        return ilastish_seg_model.predict(field, model) == ilastish_seg_model.FOREGROUND

    return xr.apply_ufunc(
        segment,
        gfp,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[bool],
    )


def run(work_dir: Path, ilastish_model_path: Path):
    experiment = xr.open_zarr(work_dir / "acquisition_data.zarr")
    model = joblib.load(ilastish_model_path)

    gfp = experiment.sel(channel="GFP")
    segmentation = neurite_seg(gfp, model)

    segmentation.to_zarr(work_dir / "neurite_seg.zarr")
