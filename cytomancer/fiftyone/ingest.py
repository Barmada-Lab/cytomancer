from more_itertools import chunked
from pathlib import Path
import logging
import uuid

from distributed import as_completed, get_client
from fiftyone import ViewField as F
from skimage import exposure  # type: ignore
from tqdm import tqdm
from PIL import Image
import xarray as xr
import fiftyone as fo
import tifffile
import dask

from cytomancer.experiment import Axes
from cytomancer.config import config
from cytomancer.io import cq1_loader

logger = logging.getLogger(__name__)


def ingest_cq1_longitudinal(experiment_path: Path):

    client = get_client()

    experiment_name = experiment_path.name
    cache_dir = config.fo_cache / experiment_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if fo.dataset_exists(experiment_name):
        raise ValueError("Dataset already exists; delete before re-ingesting")

    dataset = fo.Dataset(name=experiment_name)
    dataset.persistent = True

    df, _, _ = cq1_loader.get_experiment_df(experiment_path, ordinal_time=True)
    axes = df.index.names

    def create_sample(coord_row):
        coords, row = coord_row
        path = row["path"]
        tags_dict = dict(zip(axes, coords))

        arr = tifffile.imread(path)
        rescaled = exposure.rescale_intensity(arr, out_range="uint8")
        image = Image.fromarray(rescaled)
        png_path = cache_dir / f"{uuid.uuid4()}.png"
        image.save(png_path, format="PNG")

        return (path, png_path, tags_dict)

    chunks = list(chunked(df.iterrows(), 100))
    for chunk in tqdm(chunks):
        for _, result in as_completed(client.map(create_sample, chunk), with_results=True):
            raw_path, png_path, tags = result  # type: ignore
            sample = fo.Sample(filepath=png_path)
            sample["raw_path"] = str(raw_path)  # attach the rawpath for quantitative stuff
            for key, value in tags.items():
                sample[key.name] = value
            dataset.add_sample(sample)

    t1_view = (
        dataset
        .filter_field(Axes.TIME.name, F() == 0)
        .filter_field(Axes.FIELD.name, F() == "1")
    )
    dataset.save_view("T1 go / no-go", t1_view, color="orange")
    dataset.save()


def xarr_coords_to_tags(xarr):
    pass


def tags_to_xarr_coords(tags):
    pass


def ingest_experiment(experiment: xr.DataArray):

    experiment_name = experiment.attrs["experiment_name"]
    cache_dir = config.fo_cache / experiment_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if fo.dataset_exists(experiment_name):
        raise ValueError("Dataset already exists; delete before re-ingesting")

    dataset = fo.Dataset(name=experiment_name)
    dataset.persistent = True

    @dask.delayed
    def create_sample(frame):
        png_path = cache_dir / f"{uuid.uuid4()}.png"
        image = Image.fromarray(frame)
        image.save(png_path, format="PNG")
        tags = xarr_coords_to_tags(frame.coords)
        sample = fo.Sample(filepath=png_path, tags=tags)
        dataset.add_sample(sample)

    chunked = experiment.chunk({
        Axes.TIME: 1,
        Axes.REGION: 1,
        Axes.FIELD: 1,
        Axes.CHANNEL: 1,
    })

    assert chunked.chunks is not None

    for idx in chunked.chunks:
        print(idx)
