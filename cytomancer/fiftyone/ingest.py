from pathlib import Path
import logging
import uuid

from distributed import as_completed, get_client
from skimage import exposure  # type: ignore
from tqdm import tqdm
from PIL import Image
import fiftyone as fo
import pandas as pd
import tifffile
import dask

from cytomancer.config import config
from cytomancer.io import cq1_loader

logger = logging.getLogger(__name__)


def get_or_create_dataset(name: str) -> fo.Dataset:
    media_dir = config.fo_cache / name
    media_dir.mkdir(parents=True, exist_ok=True)

    if fo.dataset_exists(name):
        raise ValueError("Dataset already exists; delete before re-ingesting")

    dataset = fo.Dataset(name=name)
    dataset.info["media_dir"] = str(media_dir)  # type: ignore
    dataset.persistent = True
    return dataset


def ingest_experiment_df(dataset: fo.Dataset, df: pd.DataFrame):

    client = get_client()
    axes = df.index.names
    media_dir = Path(dataset.info["media_dir"])  # type: ignore

    @dask.delayed
    def prepare_sample(coord_row):
        coords, row = coord_row
        path = row["path"]
        tags_dict = dict(zip(axes, coords))

        arr = tifffile.imread(path)
        rescaled = exposure.rescale_intensity(arr, out_range="uint8")
        image = Image.fromarray(rescaled)
        png_path = media_dir / f"{uuid.uuid4()}.png"
        image.save(png_path, format="PNG")

        return (path, png_path, tags_dict)

    samples = list(map(prepare_sample, df.iterrows()))
    results = as_completed(client.compute(samples), with_results=True)
    for _, (raw_path, png_path, tags) in tqdm(results, total=len(samples)):  # type: ignore
        sample = fo.Sample(filepath=png_path)
        sample["raw_path"] = str(raw_path)  # attach the rawpath for quantitative stuff
        for key, value in tags.items():
            sample[key.name] = value
        dataset.add_sample(sample)

    dataset.save()


def ingest_cq1(base_path: Path):
    dataset = get_or_create_dataset(base_path.name)
    df = cq1_loader.get_experiment_df(base_path)
    ingest_experiment_df(dataset, df)
