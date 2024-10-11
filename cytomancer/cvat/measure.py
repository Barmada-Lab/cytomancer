from pathlib import Path
from typing import Callable
from dataclasses import dataclass
import logging
import atexit
import shutil

from pycocotools.coco import COCO
import xarray as xr
import pandas as pd
import numpy as np

from cytomancer.experiment import ExperimentType
from cytomancer.utils import load_experiment, iter_idx_prod
from cytomancer.config import config

logger = logging.getLogger(__name__)


@dataclass
class Roi:
    id: int
    label: str
    mask: np.ndarray


def broadcast_and_concat(f: Callable[[np.ndarray, np.ndarray], pd.DataFrame]) -> Callable[[np.ndarray, xr.DataArray], pd.DataFrame]:
    def wrapper(mask: np.ndarray, intensity: xr.DataArray) -> pd.DataFrame:
        measurement_df = pd.DataFrame()
        for frame in iter_idx_prod(intensity, subarr_dims=["y", "x"]):
            coords = {coord: frame.coords[coord].values for coord in frame.coords}
            frame_measurements = f(mask, frame.values).assign(**coords)  # type: ignore
            measurement_df = pd.concat([measurement_df, frame_measurements])
        return measurement_df
    return wrapper


@broadcast_and_concat
def mean_roi(mask: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    measurement_name = "mean"
    assert measurement_name in measurement_fn_lut
    return pd.DataFrame.from_records([{measurement_name: np.mean(intensity[mask])}])


@broadcast_and_concat
def median_roi(mask: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    measurement_name = "median"
    assert measurement_name in measurement_fn_lut
    return pd.DataFrame.from_records([{measurement_name: np.median(intensity[mask])}])


@broadcast_and_concat
def area_roi(mask: np.ndarray, _: np.ndarray) -> pd.DataFrame:
    measurement_name = "area"
    assert measurement_name in measurement_fn_lut
    return pd.DataFrame.from_records([{measurement_name: np.sum(mask)}])


@broadcast_and_concat
def std_roi(mask: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    measurement_name = "std"
    assert measurement_name in measurement_fn_lut
    return pd.DataFrame.from_records([{measurement_name: np.std(intensity[mask])}])


def measure(roi: Roi, intensity: xr.DataArray, measurement_names):

    roi_meta = {"roi_id": roi.id, "label": roi.label}
    dfs = []
    for measurement_name in measurement_names:
        measurement_fn = measurement_fn_lut[measurement_name]
        measurement = measurement_fn(roi.mask, intensity).assign(**roi_meta)
        dfs.append(measurement)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df)

    return merged


measurement_fn_lut = {
    'mean': mean_roi,
    'median': median_roi,
    'area': area_roi,
    'std': std_roi
}


broadcast_modes = {
    "channel",
    "z",
    "time"
}


def measure_experiment(  # noqa: C901
        experiment_dir: Path,
        experiment_type: ExperimentType,
        roi_set_name: str,
        measurement_names: set[str],
        z_projection_mode: str = "none",
        roi_broadcasting: list[str] = []):

    if (not_supported := measurement_names - measurement_fn_lut.keys()):
        raise ValueError(f"Invalid measurements: {not_supported}")

    if (not_supported := set(roi_broadcasting) - broadcast_modes):
        raise ValueError(f"Invalid broadcast mode: {not_supported}")

    if z_projection_mode not in ["none", "max", "sum"]:
        raise ValueError(f"Invalid intensity projection: {z_projection_mode}")

    logger.info("Reading experiment directory...")
    experiment = load_experiment(experiment_dir, experiment_type)

    logger.info("Caching experiment as zarray... this may take a few minutes.")
    exp_cache_dir = config.scratch_dir / (experiment_dir.name + ".zarr")
    atexit.register(lambda: shutil.rmtree(exp_cache_dir, ignore_errors=True))
    ds = xr.Dataset(dict(intensity=experiment))
    ds.to_zarr(exp_cache_dir, mode="w")

    intensity = xr.open_zarr(exp_cache_dir).intensity

    match z_projection_mode:
        case "none":
            pass
        case "mip":
            intensity = intensity.max("z")
        case "sum":
            intensity = intensity.sum("z")

    if z_projection_mode != "none" and "z" in roi_broadcasting:
        raise ValueError("Cannot broadcast over z axis with z projection enabled. Disable one or the other.")

    upload_record_location = experiment_dir / "results" / "cvat_upload.csv"
    if not upload_record_location.exists():
        raise FileNotFoundError(f"Upload record not found at {upload_record_location}! Are you sure you've uploaded using the latest version of cytomancer and provided the correct experiment folder?")

    dtype_spec = {"channel": str, "z": str, "region": str, "field": str}
    try:
        task_df = pd.read_csv(upload_record_location, dtype=dtype_spec, parse_dates=["time"]).set_index("frame")
    except ValueError:
        task_df = pd.read_csv(upload_record_location, dtype=dtype_spec).set_index("frame")

    annotations_location = experiment_dir / "results" / "annotations" / roi_set_name
    if not annotations_location.exists():
        raise FileNotFoundError(f"Annotations not found at {annotations_location}! Export annotations with 'cyto cvat export' before measuring.")

    annotations = COCO(annotations_location)

    cat_map = {id: cat["name"] for id, cat in annotations.cats.items()}

    measurements_df = pd.DataFrame()
    for img in annotations.imgs.values():
        file_name = img["file_name"]

        selector = task_df.loc[file_name].to_dict()
        for broadcast_dim in roi_broadcasting:
            selector.pop(broadcast_dim)

        subarr = intensity.sel(selector).load()
        for annotation in annotations.imgToAnns[img["id"]]:
            roi = Roi(
                id=annotation["id"],
                label=cat_map[annotation["category_id"]],
                mask=(annotations.annToMask(annotation) == 1)
            )

            if file_name not in task_df.index:
                logger.error(f"{file_name} not found in {upload_record_location}! Skipping.")
                continue

            roi_measurements = measure(roi, subarr, measurement_names)
            measurements_df = pd.concat([measurements_df, roi_measurements])

    column_order = measurements_df.columns.tolist()
    for col in measurement_names:
        column_order.remove(col)
        column_order.append(col)

    measurements_df = measurements_df.reindex(columns=column_order)

    roi_set_name_stripped = roi_set_name.replace(".json", "")
    output_dir = experiment_dir / "results" / "measurements"
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements_df.to_csv(
        output_dir / f"measurements_{roi_set_name_stripped}.csv",
        index=False,
        float_format="%.3f")
