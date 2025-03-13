import logging
from pathlib import Path

import dask
import fiftyone as fo
import pandas as pd
import tifffile
from distributed import as_completed, get_client
from PIL import Image
from skimage import exposure  # type: ignore
from tqdm import tqdm

from cytomancer.config import config
from cytomancer.io import cq1_loader

logger = logging.getLogger(__name__)


def get_or_create_dataset(name: str) -> fo.Dataset:
    media_dir = config.fo_cache / name
    media_dir.mkdir(parents=True, exist_ok=True)

    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    else:
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
        fields_dict = dict(zip(axes, coords, strict=False))

        region = fields_dict["region"]
        field = fields_dict["field"]
        z = fields_dict["z"]
        time = fields_dict["time"]
        channel = fields_dict["channel"]

        fields_dict["region_field_key"] = "-".join(map(str, [region, field, z]))
        fields_dict["time_stack_key"] = "-".join(map(str, [region, field, z, channel]))
        fields_dict["channel_stack_key"] = "-".join(
            map(
                str,
                [
                    region,
                    field,
                    z,
                    time,
                ],
            )
        )
        fields_dict["z_stack_key"] = "-".join(
            map(
                str,
                [
                    region,
                    field,
                    time,
                    channel,
                ],
            )
        )

        arr = tifffile.imread(path)
        rescaled = exposure.rescale_intensity(arr, out_range="uint8")
        image = Image.fromarray(rescaled)
        filename = f"T{time}_C{channel}_R{region}_F{field}_Z{z}.png"
        png_path = media_dir / filename
        image.save(png_path, format="PNG")

        return (path, png_path, fields_dict)

    samples = list(map(prepare_sample, df.iterrows()))
    results = as_completed(client.compute(samples), with_results=True)
    for _, (raw_path, png_path, tags) in tqdm(results, total=len(samples)):  # type: ignore
        sample = fo.Sample(filepath=png_path)
        sample["raw_path"] = str(raw_path)  # attach the rawpath for quantitative stuff
        for key, value in tags.items():
            sample[key] = value
        dataset.add_sample(sample)

    timeseries_view = dataset.group_by("time_stack_key", order_by="time")
    channel_stack_view = dataset.group_by("channel_stack_key", order_by="channel")
    z_stack_view = dataset.group_by("z_stack_key", order_by="z")

    dataset.save_view("timeseries", timeseries_view)
    dataset.save_view("channel_stacks", channel_stack_view)
    dataset.save_view("z_stacks", z_stack_view)

    dataset.save()


def do_ingest_cq1(
    base_path: Path,
    name: str,
    regions: list[str],
    fields: list[str],
    channels: list[str],
    timepoints: list[int],
):
    dataset = get_or_create_dataset(name)
    df = cq1_loader.get_experiment_df(base_path).reset_index()

    if regions:
        df = df[df["region"].isin(regions)]

    if fields:
        df = df[df["field"].isin(fields)]

    if channels:
        df = df[df["channel"].isin(channels)]

    if timepoints:
        df = df[df["timepoint"].isin(timepoints)]

    df = df.set_index(["region", "field", "timepoint", "time", "channel", "z"])

    ingest_experiment_df(dataset, df)
