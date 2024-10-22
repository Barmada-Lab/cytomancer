import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr

logger = logging.getLogger(__name__)


META_JSON = "meta.json"
SUMMARY_CSV = "summary.csv"


@dataclass
class CytoMeta:
    shape: tuple
    dtype: str

    def dump_json(self, path: Path):
        with path.open("w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load_json(cls, path: Path):
        return CytoMeta(**json.loads(path.read_text()))


def _get_experiment_df(csv_path: Path):
    assert csv_path.exists(), f"File not found: {csv_path}, is this a valid cyto dir?"
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])

    preliminary_mi = pd.MultiIndex.from_frame(df.drop(["path"], axis=1))
    holy_mi = pd.MultiIndex.from_product(
        preliminary_mi.levels, names=preliminary_mi.names
    )
    holy_df = (
        df[["path"]]
        .set_index(preliminary_mi)
        .reindex(index=holy_mi)
        .sort_index()
        .replace({np.nan: None})
    )

    return holy_df


def _get_experiment_meta(meta_path: Path):
    assert meta_path.exists(), f"File not found: {meta_path}, is this a valid cyto dir?"
    meta = CytoMeta.load_json(meta_path)
    return meta


def _load_df(df, meta: CytoMeta):
    def read_img(path):
        logger.debug(f"Reading {path}")
        if path is None:
            logger.warning(
                "MeasurementResult.ome.xml is missing an image. This is likely the result of an acquisition error! Replacing with NaNs..."
            )
            return np.full(meta.shape, np.nan)
        elif not path.exists():
            logger.warning(
                f"Could not find image at {path}, even though its existence is recorded in MeasurementResult.ome.xml. The file may have been moved or deleted. Replacing with NaNs..."
            )
            return np.full(meta.shape, np.nan)
        return tifffile.imread(path).astype(meta.dtype)

    def read_indexed_ims(recurrence):
        """
        Recursively read and stack images from a sorted hierarchical index
        in a breadth-first manner.
        """
        if type(recurrence) is pd.Series:
            path = recurrence["path"]
            return da.from_delayed(
                dask.delayed(read_img)(path), meta.shape, dtype=meta.dtype
            )
        else:  # type(recurrence) is pd.DataFrame
            if type(recurrence.index) is pd.MultiIndex:
                level = recurrence.index.levels[0]  # type: ignore
            else:  # type(recurrence.index) is pd.Index
                level = recurrence.index.values
            return da.stack([read_indexed_ims(recurrence.loc[idx]) for idx in level])

    arr = read_indexed_ims(df)
    labels = df.index.names
    arr = xr.DataArray(
        arr,
        dims=labels + ["y", "x"],
        coords=dict(zip(labels, df.index.levels, strict=False)),
    )  # type: ignore

    arr.coords["channel"] = arr.coords["channel"].astype(str)
    arr.coords["region"] = arr.coords["region"].astype(str)
    arr.coords["field"] = arr.coords["field"].astype(str)

    if "z" in arr.dims:
        arr = arr.squeeze("z", drop=True)
    return arr


def load_dir(
    img_dir: Path, summary_csv: Path | None = None, meta_json: Path | None = None
):
    if summary_csv is None:
        summary_csv = img_dir / SUMMARY_CSV

    if meta_json is None:
        meta_json = img_dir / META_JSON

    df = _get_experiment_df(summary_csv)
    df["path"] = df["path"].apply(lambda x: img_dir / x)
    meta = _get_experiment_meta(meta_json)

    return _load_df(df, meta)
