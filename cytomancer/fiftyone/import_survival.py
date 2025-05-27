from pathlib import Path

import fiftyone as fo
import numpy as np
import xarray as xr
from pandas import Timestamp
from skimage.measure import regionprops
from tqdm import tqdm

from cytomancer.utils import iter_idx_prod


def add_detection_results(sample: fo.Sample, labels: np.ndarray, preds: np.ndarray):
    detections = []
    for props in regionprops(labels):
        mask = labels == props.label
        prediction = np.bincount(preds[mask]).argmax()
        pred_label = "live" if prediction == 1 else "dead"
        detection = fo.Detection.from_mask(mask, label=pred_label)
        detections.append(detection)
    sample["predictions"] = fo.Detections(detections=detections)
    return sample


def import_survival(dataset: fo.Dataset, survival_results: xr.Dataset):
    n_samples = (
        survival_results.sizes["region"]
        * survival_results.sizes["field"]
        * survival_results.sizes["time"]
    )
    for frame in tqdm(
        iter_idx_prod(survival_results, subarr_dims=["y", "x"]), total=n_samples
    ):
        selector = {coord: frame[coord].values.tolist() for coord in frame.coords}
        selector["time"] = Timestamp(selector["time"], unit="ns")
        preds = frame["preds"].values
        labels = frame["nuc_labels"].values
        for match in dataset.match(selector):
            add_detection_results(match, labels, preds).save()


def do_import_survival(experiment_dir: Path, dataset_name: str):
    dataset = fo.load_dataset(dataset_name)
    survival_results = xr.open_zarr(
        experiment_dir / "results" / "survival_processed.zarr"
    )
    import_survival(dataset, survival_results)
