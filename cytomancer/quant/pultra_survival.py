import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Worker, get_client
from skimage.measure import regionprops
from sklearn.pipeline import Pipeline

from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.io.cyto_dir import load_dir
from cytomancer.utils import load_experiment

from .pultra_classifier import load_classifier

logger = logging.getLogger(__name__)

LIVE = 1
DEAD = 2

DAPI_SNR_THRESHOLD = 2


def get_features(mask, dapi, gfp, rfp, field_medians):
    dapi_signal = dapi[mask].mean() / field_medians[0]
    gfp_signal = gfp[mask].mean() / field_medians[1]
    rfp_signal = rfp[mask].mean() / field_medians[2]
    size = mask.astype(int).sum()
    return {
        "dapi_signal": dapi_signal,
        "gfp_signal": gfp_signal,
        "rfp_signal": rfp_signal,
        "size": size,
    }


def predict(dapi, gfp, rfp, nuc_labels, classifier):
    dapi_field_med = np.median(dapi)
    gfp_field_med = np.median(gfp)
    rfp_field_med = np.median(rfp)

    preds = np.zeros_like(nuc_labels, dtype=np.uint8)
    for props in regionprops(nuc_labels):
        mask = nuc_labels == props.label
        dapi_mean = dapi[mask].mean()

        # filter dim objects
        if dapi_mean / dapi_field_med < DAPI_SNR_THRESHOLD:
            continue

        features = get_features(
            mask, dapi, gfp, rfp, [dapi_field_med, gfp_field_med, rfp_field_med]
        )
        df = pd.DataFrame.from_records([features])
        if classifier.predict(df)[0]:
            preds[mask] = LIVE
        else:
            preds[mask] = DEAD

    return preds


def process(intensity: xr.DataArray, nuc_labels: xr.DataArray, classifier: Pipeline):
    def process_field(
        dapi: np.ndarray, gfp: np.ndarray, rfp: np.ndarray, nuc_labels: np.ndarray
    ):
        if np.issubdtype(nuc_labels.dtype, np.floating):
            return np.full_like(dapi, np.iinfo(np.uint8).max, dtype=np.uint8)

        if np.isnan(dapi).any() or np.isnan(gfp).any() or np.isnan(rfp).any():
            return np.full_like(dapi, np.iinfo(np.uint8).max, dtype=np.uint8)

        preds = predict(dapi, gfp, rfp, nuc_labels, classifier)
        return preds

    preds = xr.apply_ufunc(
        process_field,
        intensity.sel(channel="DAPI").drop_vars("channel"),
        intensity.sel(channel="GFP").drop_vars("channel"),
        intensity.sel(channel="RFP").drop_vars("channel"),
        nuc_labels.sel(channel="DAPI").drop_vars("channel"),
        input_core_dims=[["y", "x"], ["y", "x"], ["y", "x"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        join="inner",
        output_dtypes=[np.uint8],
    )

    return xr.Dataset({"preds": preds})


def quantify(nuc_labels: xr.DataArray, preds: xr.DataArray):
    def quantify_field(nuc_labels, preds):
        if (nuc_labels == np.iinfo(np.uint16).max).all():
            return np.asarray([np.nan])
        return np.asarray(
            [np.unique(nuc_labels[np.where(preds == LIVE)]).shape[0]], dtype=float
        )

    return xr.apply_ufunc(
        quantify_field,
        nuc_labels.sel(channel="DAPI").drop_vars("channel"),
        preds,
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["count"]],
        dask_gufunc_kwargs={"output_sizes": {"count": 1}},
        vectorize=True,
        dask="parallelized",
    ).squeeze("count", drop=True)


def run(
    experiment_path: Path,
    experiment_type: ExperimentType,
    svm_model_path: Path,
    save_annotations: bool,
):
    results_dir = experiment_path / "results"
    seg_results_dir = results_dir / "stardist_nuc_seg"
    assert (
        seg_results_dir.exists()
    ), "Nuclear segmentation results not found. Please run nuclear segmentation first."

    client = get_client()
    logger.info(f"Connected to dask scheduler {client.scheduler}")
    logger.info(f"Dask dashboard available at {client.dashboard_link}")
    logger.debug(f"Cluster: {client.cluster}")
    logger.info(f"Starting analysis of {experiment_path}")

    def init_logging(dask_worker: Worker):
        fmt = f"{dask_worker.id}|%(asctime)s|%(name)s|%(levelname)s: %(message)s"
        # disable GPU for workers. Although stardist is GPU accelerated, it's
        # faster to run many CPU workers in parallel
        logging.basicConfig(level=config.log_level, format=fmt)
        logging.getLogger("dask").setLevel(level=logging.WARN)
        logging.getLogger("distributed.nanny").setLevel(level=logging.WARN)
        logging.getLogger("distributed.scheduler").setLevel(level=logging.WARN)
        logging.getLogger("distributed.core").setLevel(level=logging.WARN)
        logging.getLogger("distributed.http").setLevel(level=logging.WARN)

    client.register_worker_callbacks(init_logging)

    nuc_labels = load_dir(seg_results_dir).isel(y=slice(0, 1998), x=slice(0, 1998))
    intensity = load_experiment(experiment_path, experiment_type)

    logger.debug(f"loading classifier from {svm_model_path}")
    if (classifier := load_classifier(svm_model_path)) is None:
        raise ValueError(f"Could not load classifier model at path {svm_model_path}")

    preds = process(intensity, nuc_labels, classifier)

    print(preds)
    if save_annotations:
        store_path = results_dir / "survival_processed.zarr"
        preds["nuc_labels"] = nuc_labels
        preds.to_zarr(store_path, mode="w")
        preds = xr.open_zarr(store_path)

    quantified = quantify(nuc_labels, preds["preds"])
    df = quantified.to_dataframe(
        name="count", dim_order=["region", "field", "time"]
    ).dropna()

    df["count"] = df["count"].astype(int)
    df.to_csv(results_dir / "survival.csv")

    logger.info(f"Finished analysis of {experiment_path}")
