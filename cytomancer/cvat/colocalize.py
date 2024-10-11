from itertools import product
from collections import defaultdict
from pathlib import Path
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


def find_coloc(masks_1, masks_2):
    """
    Compute colocalization of two lists of masks, yielding tuples of indices
    and the intersection of their corresponding masks
    """
    # TODO: this output of this function describes a graph. Maybe formalize that yeah?

    yielded = set()
    for (i, mask_1), (j, mask_2) in product(enumerate(masks_1), enumerate(masks_2)):
        intersection = mask_1 & mask_2
        if np.any(intersection) and (i, j) not in yielded:  # avoid duplicates
            yield ((i, j), intersection)
            yielded.add((i, j))


def find_max_coloc(masks_1, masks_2):
    """
    Find 1-1 colocalization mapping between masks in masks_1 and masks_2, maximizing
    intersection area. Yields tuples of indices and the corresponding intersection.
    """

    colocalization = defaultdict(list)
    for (i, j), intersection in find_coloc(masks_1, masks_2):
        colocalization[i].append((j, intersection))

    for i, colocs in colocalization.items():
        max_overlap = max(colocs, key=lambda x: x[1].sum())
        yield ((i, max_overlap[0]), max_overlap[1])


def group_rois(file_coords: pd.DataFrame, coco_set: COCO, variable_dims: list[str]):
    """
    Groups coco annotations by their coordinates in file_coords, yielding tuples of
    coordinates and their corresponding annotations
    """
    filename_to_anns = {coco_set.imgs[id]["file_name"]: anns for id, anns in coco_set.imgToAnns.items()}
    stripped = file_coords.drop(columns=variable_dims)
    coords = [coord for coord in stripped.columns if coord != 'frame']
    rois = stripped.groupby(coords)
    for roi_coords, df in rois:
        annotations = sum(df["frame"].apply(lambda x: filename_to_anns[x] if x in filename_to_anns else []), [])
        yield dict(zip(coords, roi_coords)), annotations


def nuc_cyto(
        intensity: xr.DataArray,
        task_df: pd.DataFrame,
        annotations: COCO,
        nuc_label: str,
        soma_label: str):

    cat_map = {id: cat["name"] for id, cat in annotations.cats.items()}

    cell_id = 1
    if nuc_label not in cat_map.values():
        raise ValueError(f"Category {nuc_label} not found in annotations!")

    if soma_label not in cat_map.values():
        raise ValueError(f"Category {soma_label} not found in annotations!")

    measurements = []
    for coords, group in group_rois(task_df, annotations, ["channel"]):
        nuc_masks = [annotations.annToMask(ann) == 1 for ann in group if cat_map[ann["category_id"]] == nuc_label]
        soma_masks = [annotations.annToMask(ann) == 1 for ann in group if cat_map[ann["category_id"]] == soma_label]

        subarr = intensity.sel(coords).load()

        for (nuc_i, soma_i), intersection in find_max_coloc(nuc_masks, soma_masks):
            nuc = nuc_masks[nuc_i]
            soma = soma_masks[soma_i]
            cyto = np.bitwise_xor(soma, intersection)

            for frame in iter_idx_prod(subarr, subarr_dims=["y", "x"]):
                nuc_mean = frame.values[nuc].mean()
                nuc_std = frame.values[nuc].std()
                soma_mean = frame.values[soma].mean()
                soma_std = frame.values[soma].std()
                cyto_mean = frame.values[cyto].mean()
                cyto_std = frame.values[cyto].std()
                frame_coords = {k: v.values.tolist() for k, v in frame.coords.items()}
                measurement = {
                    "cell_id": cell_id,
                    "nuc_mean": nuc_mean,
                    "nuc_std": nuc_std,
                    "soma_mean": soma_mean,
                    "soma_std": soma_std,
                    "cyto_mean": cyto_mean,
                    "cyto_std": cyto_std,
                    **frame_coords
                }
                measurements.append(measurement)

            cell_id += 1

    return pd.DataFrame.from_records(measurements)


def do_nuc_cyto(  # noqa: C901
        experiment_dir: Path,
        experiment_type: ExperimentType,
        roi_set_name: str,
        nuc_label: str,
        soma_label: str):

    logger.info("Reading experiment directory...")
    experiment = load_experiment(experiment_dir, experiment_type)

    logger.info("Caching experiment as zarray... this may take a few minutes.")
    exp_cache_dir = config.scratch_dir / (experiment_dir.name + ".zarr")
    atexit.register(lambda: shutil.rmtree(exp_cache_dir, ignore_errors=True))
    ds = xr.Dataset(dict(intensity=experiment))
    ds.to_zarr(exp_cache_dir, mode="w")

    intensity = xr.open_zarr(exp_cache_dir).intensity

    upload_record_location = experiment_dir / "results" / "cvat_upload.csv"
    if not upload_record_location.exists():
        raise FileNotFoundError(f"Upload record not found at {upload_record_location}! Are you sure you've uploaded using the latest version of cytomancer and provided the correct experiment folder?")

    dtype_spec = {"channel": str, "z": str, "region": str, "field": str}
    try:
        task_df = pd.read_csv(upload_record_location, dtype=dtype_spec, parse_dates=["time"])
    except ValueError:
        task_df = pd.read_csv(upload_record_location, dtype=dtype_spec)

    annotations_location = experiment_dir / "results" / "annotations" / roi_set_name
    if not annotations_location.exists():
        raise FileNotFoundError(f"Annotations not found at {annotations_location}! Export annotations with 'cyto cvat export' before measuring.")

    annotations = COCO(annotations_location)

    df = nuc_cyto(intensity, task_df, annotations, nuc_label, soma_label)

    measurement_output = experiment_dir / "results" / "measurements"
    measurement_output.mkdir(parents=True, exist_ok=True)
    df.to_csv(measurement_output / "cvat_nuc_cyto.csv", index=False)
