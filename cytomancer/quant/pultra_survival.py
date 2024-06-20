from pathlib import Path
import logging
import os

from sklearn.pipeline import Pipeline
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity
from skimage import filters, exposure, morphology  # type: ignore
from dask.distributed import Worker, get_client
import fiftyone as fo
from fiftyone import ViewField as F
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

    preds = np.zeros_like(nuc_labels)
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


def quantify(intensity: xr.DataArray, seg_model, classifier: Pipeline, dataset: fo.Dataset | None = None):

    def quantify_field(field: xr.DataArray):

        dapi = field.sel({Axes.CHANNEL: "DAPI"}).squeeze(drop=True).values
        gfp = field.sel({Axes.CHANNEL: "GFP"}).squeeze(drop=True).values
        rfp = field.sel({Axes.CHANNEL: "RFP"}).squeeze(drop=True).values

        footprint = morphology.disk(5)
        dapi_eqd = np.array([
            exposure.equalize_adapthist(
                filters.rank.median(
                    rescale_intensity(
                        frame,
                        out_range="uint8"),
                    footprint),
                kernel_size=100,
                clip_limit=0.01) for frame in dapi])
        nuc_labels = np.array([seg_model.predict_instances(frame)[0] for frame in dapi_eqd]).astype(np.uint16)  # type: ignore

        preds = np.array([predict(dapi_frame, gfp_frame, rfp_frame, nuc_label_frame, classifier) for dapi_frame, gfp_frame, rfp_frame, nuc_label_frame in zip(dapi, gfp, rfp, nuc_labels)]).astype(np.uint8)

        counts = []
        for idx, (label_frame, pred_frame) in enumerate(zip(nuc_labels, preds)):

            count = 0
            detections = []
            for props in regionprops(label_frame):
                mask = label_frame == props.label

                prediction = np.argmax(np.bincount(pred_frame[mask]))
                if prediction == LIVE:
                    label = "live"
                    count += 1
                elif prediction == DEAD:
                    label = "dead"
                else:  # censored
                    continue

                if dataset is not None:
                    detections.append(
                        fo.Detection.from_mask(mask, label=label))

            if dataset is not None:
                time = idx
                region_id = str(field.squeeze()[Axes.REGION].values)
                field_id = str(field.squeeze()[Axes.FIELD].values)
                logger.info(f"Processing time {time}, region {region_id}, field {field_id}: {count} live cells detected")
                dapi_sample = (
                    dataset
                    .filter_field(Axes.TIME.name, F() == time)
                    .filter_field(Axes.REGION.name, F() == region_id)
                    .filter_field(Axes.FIELD.name, F() == field_id)
                    .filter_field(Axes.CHANNEL.name, F() == "DAPI")
                    .first())
                dapi_sample["predictions"] = fo.Detections(detections=detections)
                dapi_sample.save()

            counts.append(count)

        return xr.DataArray(
            data=np.array(counts).reshape(-1, 1, 1),
            dims=[Axes.TIME, Axes.REGION, Axes.FIELD],
            coords={Axes.TIME: field[Axes.TIME], Axes.REGION: field[Axes.REGION], Axes.FIELD: field[Axes.FIELD]})

    chunked = intensity.chunk({
        Axes.TIME: -1,
        Axes.CHANNEL: -1,
        Axes.REGION: 1,
        Axes.FIELD: 1
    })

    # final result is cellcount as a fuction of time, region, and field
    template = chunked.isel({
        Axes.CHANNEL: 0,
        Axes.X: 0,
        Axes.Y: 0
    }).drop_vars(Axes.CHANNEL).squeeze(drop=True)

    return xr.map_blocks(quantify_field, chunked, template=template)


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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.basicConfig(level=config.log_level, format=fmt)
        logging.getLogger("dask").setLevel(level=logging.WARN)
        logging.getLogger("distributed.nanny").setLevel(level=logging.WARN)
        logging.getLogger("distributed.scheduler").setLevel(level=logging.WARN)
        logging.getLogger("distributed.core").setLevel(level=logging.WARN)
        logging.getLogger("distributed.http").setLevel(level=logging.WARN)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(3)

    client.register_worker_callbacks(init_logging)

    logger.debug(f"loading classifier from {svm_model_path}")
    if (classifier := load_classifier(svm_model_path)) is None:
        raise ValueError(f"Could not load classifier model at path {svm_model_path}")

    intensity = load_experiment(experiment_path, experiment_type)

    if experiment_type is ExperimentType.CQ1:
        # The last row/col of DAPI images captured on the CQ1 is inexplicably zeroed out.
        # We slice it out to avoid issues further down the line
        intensity = intensity.isel({Axes.X: slice(0, 1998), Axes.Y: slice(0, 1998)})

    results_dir = experiment_path / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    dataset = None
    if save_annotations and fo.dataset_exists(experiment_path.name):
        dataset = fo.load_dataset(experiment_path.name)
    elif save_annotations and not fo.dataset_exists(experiment_path.name):
        logger.warn(f"Could not find dataset for {experiment_path.name}; did you run fiftyone ingest on your experiment? Annotations will not be saved.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    assert model is not None, "Could not load stardist model"

    (
        quantify(intensity, model, classifier, dataset)
        .rename({
            Axes.TIME: "time",
            Axes.REGION: "well",
            Axes.FIELD: "field"
        })
        .to_dataframe("count")
        .to_csv(results_dir / "survival.csv", index=False)
    )

    logger.info(f"Finished analysis of {experiment_path}")
