import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from acquisition_io import ExperimentType
from skimage.morphology import skeletonize as skimage_skeletonize

from cytomancer.config import config
from cytomancer.io.cyto_dir import CytoMeta
from cytomancer.io.utils import stage_experiment
from cytomancer.models.unet.predict import load_unet, predict
from cytomancer.utils import iter_idx_prod

logger = logging.getLogger(__name__)


def quantify(path: Path, skeletonize: bool) -> int:
    segmentation = tifffile.imread(path)
    if skeletonize:
        return skimage_skeletonize(segmentation).sum()
    else:
        return segmentation.sum()


def run_experiment(
    experiment_path: Path,
    experiment_type: ExperimentType,
    model_path: Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = load_unet(model_path, device=device)
    dataset = stage_experiment(experiment_path, experiment_type)

    dataset = dataset.sel(channel="GFP")

    neurite_seg_dir = experiment_path / "scratch" / "neurite_seg"
    neurite_seg_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for arr in iter_idx_prod(dataset.intensity, ["y", "x"]):
        mask = predict(model, arr.values, threshold=0.5, device=device)
        region = str(arr.coords["region"].values)
        field = str(arr.coords["field"].values)
        time = int(arr.coords["time"].values)
        filename = f"{region}_{field}_{time}.tif"
        tifffile.imwrite(
            neurite_seg_dir / filename, mask.astype(np.uint8), compression="lzw"
        )
        coords = {
            "region": region,
            "field": field,
            "time": time,
            "channel": "GFP",
            "path": filename,
        }
        records.append(coords)

    output_df = pd.DataFrame.from_records(records)
    output_df.to_csv(neurite_seg_dir / "summary.csv", index=False)
    CytoMeta((arr.sizes["y"], arr.sizes["x"]), "uint8").dump_json(
        neurite_seg_dir / "meta.json"
    )

    summary = pd.read_csv(neurite_seg_dir / "summary.csv").drop("channel", axis=1)
    summary = summary.sort_values(by=["region", "field", "time"])

    with Pool(processes=config.dask_n_workers) as p:
        paths = summary["path"].map(lambda x: neurite_seg_dir / x).tolist()
        summary["skeleton_length"] = p.map(partial(quantify, skeletonize=True), paths)
        summary["mask_area"] = p.map(partial(quantify, skeletonize=False), paths)
        summary.pop("path")

    analyzis_dir = experiment_path / "analysis"
    analyzis_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(analyzis_dir / "neurite_quant.csv", index=False)
