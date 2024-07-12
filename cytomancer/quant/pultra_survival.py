from pathlib import Path
import logging
import os

from sklearn.pipeline import Pipeline
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage import filters, exposure, morphology  # type: ignore
from dask.distributed import Worker, get_client
from fiftyone import ViewField as F
import fiftyone as fo
import xarray as xr
import pandas as pd
import numpy as np

from cytomancer.utils import load_experiment
from cytomancer.config import config
from cytomancer.experiment import ExperimentType, Axes
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
        "size": size
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

        features = get_features(mask, dapi, gfp, rfp, [dapi_field_med, gfp_field_med, rfp_field_med])
        df = pd.DataFrame.from_records([features])
        if classifier.predict(df)[0]:
            preds[mask] = LIVE
        else:
            preds[mask] = DEAD

    return preds


def process(intensity: xr.DataArray, seg_model, classifier: Pipeline):

    def process_field(dapi: np.ndarray, gfp: np.ndarray, rfp: np.ndarray):

        if np.isnan(dapi).any() or np.isnan(gfp).any() or np.isnan(rfp).any():
            return (
                np.full_like(dapi, np.iinfo(np.uint16).max, dtype=np.uint16),
                np.full_like(dapi, np.iinfo(np.uint8).max, dtype=np.uint8)
            )

        footprint = morphology.disk(5)

        rescaled = rescale_intensity(dapi, out_range="uint8")
        med = filters.rank.median(rescaled, footprint)
        eqd = exposure.equalize_adapthist(med, kernel_size=100, clip_limit=0.01)
        nuc_labels = seg_model.predict_instances(eqd)[0].astype(np.uint16)  # type: ignore
        preds = predict(dapi, gfp, rfp, nuc_labels, classifier)

        return nuc_labels, preds

    nuc_labels, preds = xr.apply_ufunc(
        process_field,
        intensity.sel({Axes.CHANNEL: "DAPI"}).drop_vars(Axes.CHANNEL),
        intensity.sel({Axes.CHANNEL: "GFP"}).drop_vars(Axes.CHANNEL),
        intensity.sel({Axes.CHANNEL: "RFP"}).drop_vars(Axes.CHANNEL),
        input_core_dims=[[Axes.Y, Axes.X], [Axes.Y, Axes.X], [Axes.Y, Axes.X]],
        output_core_dims=[[Axes.Y, Axes.X], [Axes.Y, Axes.X]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint16, np.uint8])

    return xr.Dataset({
        "nuc_labels": nuc_labels,
        "preds": preds
    })


def quantify(results: xr.Dataset):

    def quantify_field(nuc_labels, preds):
        if (nuc_labels == np.iinfo(np.uint16).max).all():
            return np.asarray([np.nan])
        return np.asarray([np.unique(nuc_labels[np.where(preds == LIVE)]).shape[0]], dtype=float)

    return xr.apply_ufunc(
        quantify_field,
        results["nuc_labels"],
        results["preds"],
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["count"]],
        dask_gufunc_kwargs={"output_sizes": {"count": 1}},
        vectorize=True,
        dask="parallelized").squeeze("count", drop=True)


def run(
        experiment_path: Path,
        experiment_type: ExperimentType,
        svm_model_path: Path,
        save_annotations: bool = False):

    client = get_client()
    logger.info(f"Connected to dask scheduler {client.scheduler}")
    logger.info(f"Dask dashboard available at {client.dashboard_link}")
    logger.debug(f"Cluster: {client.cluster}")
    logger.info(f"Starting analysis of {experiment_path}")

    def init_logging(dask_worker: Worker):
        fmt = f"{dask_worker.id}|%(asctime)s|%(name)s|%(levelname)s: %(message)s"
        # disable GPU for workers. Although stardist is GPU accelerated, it's
        # faster to run many CPU workers in parallel
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.basicConfig(level=config.log_level, format=fmt)
        logging.getLogger("dask").setLevel(level=logging.WARN)
        logging.getLogger("distributed.nanny").setLevel(level=logging.WARN)
        logging.getLogger("distributed.scheduler").setLevel(level=logging.WARN)
        logging.getLogger("distributed.core").setLevel(level=logging.WARN)
        logging.getLogger("distributed.http").setLevel(level=logging.WARN)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.get_logger().setLevel('ERROR')

    client.register_worker_callbacks(init_logging)

    logger.debug(f"loading classifier from {svm_model_path}")
    if (classifier := load_classifier(svm_model_path)) is None:
        raise ValueError(f"Could not load classifier model at path {svm_model_path}")

    intensity = load_experiment(experiment_path, experiment_type)

    results_dir = experiment_path / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    dataset = None
    if save_annotations and fo.dataset_exists(experiment_path.name):
        dataset = fo.load_dataset(experiment_path.name)
    elif save_annotations and not fo.dataset_exists(experiment_path.name):
        logger.warn(f"Could not find dataset for {experiment_path.name}; did you run fiftyone ingest on your experiment? Annotations will not be saved.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.get_logger().setLevel('ERROR')
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    assert model is not None, "Could not load stardist model"

    intensity = intensity.sel({Axes.REGION: ["C04"]})
    store_path = results_dir / "survival_processed.zarr"
    process(intensity, model, classifier).to_zarr(store_path, mode="w")

    df = (
        quantify(xr.open_zarr(store_path))
        .to_dataframe(name="count", dim_order=["region", "field", "time"])
        .dropna()
    )
    df["count"] = df["count"].astype(int)
    df.to_csv(results_dir / "survival.csv")

    logger.info(f"Finished analysis of {experiment_path}")
