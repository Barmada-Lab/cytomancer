import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import tifffile
from distributed import get_client, wait
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import rank
from skimage.morphology import disk
from stardist.models import StarDist2D

from cytomancer.io.cq1_loader import get_experiment_df_detailed
from cytomancer.io.cyto_dir import CytoMeta

logger = logging.getLogger(__name__)


def run(
    experiment_dir: Path,
    model_name: str = "2D_versatile_fluo",
    clahe_clip: float = 0.01,
):  # noqa: C901
    """
    Run StarDist nuclear segmentation on a DataFrame of images.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of images to segment.
    model : str
        Path to the StarDist model to use.
    clahe_clip : float
        CLAHE clip limit.

    Returns
    -------
    pd.DataFrame
        DataFrame with segmentation results.
    """
    if not experiment_dir.exists():
        print(f"Could not find experiment directory at {experiment_dir}")
        return

    scratch_subdir = experiment_dir / "scratch" / "stardist_nuc_seg"
    scratch_subdir.mkdir(parents=True, exist_ok=True)
    if any(scratch_subdir.glob("*.tif")):
        for file in scratch_subdir.glob("*.tif"):
            file.unlink()

    if (model := StarDist2D.from_pretrained(model_name)) is None:
        raise ValueError(f"Failed to load model {model_name}")

    client = get_client()

    def load_and_preprocess(path):
        if path is None:
            logger.warning(
                "MeasurementResult.ome.xml is missing an image. This is likely the result of an acquisition error! Replacing with NaNs..."
            )
            return np.full((1, 1), np.nan)
        elif not path.exists():
            logger.warning(
                f"Could not find image at {path}, even though its existence is recorded in MeasurementResult.ome.xml. The file may have been moved or deleted. Replacing with NaNs..."
            )
            return np.full((1, 1), np.nan)
        image = tifffile.imread(path).astype(np.float16)[:1998, :1998]

        footprint = disk(5)
        rescaled = rescale_intensity(image, out_range="uint8")
        med = rank.median(rescaled, footprint=footprint)
        equalized = equalize_adapthist(med, clip_limit=clahe_clip)
        return equalized

    df, shape, attrs = get_experiment_df_detailed(experiment_dir)
    df = df.reset_index()
    rows = [row for _, row in df[df["channel"] == "DAPI"].iterrows()]

    summary = []
    preprocessing: dict = {}
    while len(preprocessing) > 0 or len(rows) > 0:
        while len(preprocessing) < 100 and len(rows) > 0:
            coords = rows.pop()
            path = coords.pop("path")
            future = client.submit(load_and_preprocess, path)
            preprocessing[future] = coords

        done, _ = wait(list(preprocessing.keys()), return_when="FIRST_COMPLETED")  # type: ignore
        for future in done:
            img = future.result()
            coords = preprocessing.pop(future)

            if np.isnan(img).any():
                continue

            predictions, _ = model.predict_instances(img)  # type: ignore
            filename = uuid4().hex + ".tif"
            tifffile.imwrite(scratch_subdir / filename, predictions, compression="lzw")
            row_vals = coords.tolist() + [filename]
            summary.append(row_vals)

    row_keys = df.columns.tolist()
    summary_records = [
        dict(zip(row_keys, row_vals, strict=False)) for row_vals in summary
    ]
    output_df = pd.DataFrame.from_records(summary_records)
    output_df.to_csv(scratch_subdir / "summary.csv", index=False)
    CytoMeta(shape, "uint16").dump_json(scratch_subdir / "meta.json")
