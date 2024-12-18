import atexit
import logging
import pathlib as pl
import shutil

import click
import numpy as np
import pandas as pd
import xarray as xr
from cvat_sdk import Client
from skimage.measure import regionprops

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.utils import load_experiment

from .helpers import get_project, new_client_from_config

logger = logging.getLogger(__name__)

FIELD_DELIM = "|"
FIELD_VALUE_DELIM = "-"
VALUE_DELIM = ":"


def rle_to_mask(rle: list[int], width: int, height: int) -> np.ndarray:
    assert sum(rle) == width * height, "RLE does not match image size"

    decoded = [0] * (width * height)  # create bitmap container
    decoded_idx = 0
    value = 0

    for v in rle:
        decoded[decoded_idx : decoded_idx + v] = [value] * v
        decoded_idx += v
        value = abs(value - 1)

    decoded = np.array(decoded, dtype=bool)
    decoded = decoded.reshape((height, width))  # reshape to image size
    return decoded


def shape_to_mask(shape, height, width):
    rle = list(map(int, shape.points))
    left, top, right, bottom = rle[-4:]
    patch_height, patch_width = (bottom - top + 1, right - left + 1)
    patch = rle_to_mask(rle[:-4], patch_width, patch_height)
    mask = np.zeros((height, width), dtype=bool)
    mask[top : bottom + 1, left : right + 1][patch] = True
    return mask


def get_obj_arr_and_labels(anno_table, length, height, width):
    obj_arr = np.zeros((length, height, width), dtype="uint16")
    label_arr = np.zeros((length, height, width), dtype="uint16")
    for shape in anno_table.shapes:
        obj_id = shape.id
        label_id = shape.label_id
        frame = shape.frame
        patch_mask = shape_to_mask(shape, height, width)
        obj_arr[frame][patch_mask] = obj_id
        label_arr[frame][patch_mask] = label_id
    return obj_arr, label_arr


def _parse_field_selector(selector: str):
    tokens = selector.split(FIELD_VALUE_DELIM)
    axis = tokens[0]

    field_values = FIELD_VALUE_DELIM.join(
        tokens[1:]
    )  # this allows field values to contain the FIELD_VALUE_DELIM, as is sometimes the case with filenames

    target_dtype = np.str_

    match axis:
        case "time":
            field_value_tokens = np.array(
                [np.datetime64(int(ts), "ns") for ts in field_values.split(VALUE_DELIM)]
            )
            if field_value_tokens.size == 1:
                field_value = field_value_tokens[0]
                return (axis, field_value)
            else:
                return (axis, field_value_tokens)
        case _:
            field_value_tokens = np.array(field_values.split(VALUE_DELIM)).astype(
                target_dtype
            )
            if field_value_tokens.size == 1:
                field_value = field_value_tokens[0]
                return (axis, field_value)
            else:
                return (axis, field_value_tokens)


def parse_selector(selector_str: str) -> dict[str, np.ndarray]:
    """Parses a selector string into a dictionary of axes to values"""
    return dict(map(_parse_field_selector, selector_str.split(FIELD_DELIM)))  # type: ignore


def enumerate_rois(client: Client, project_id: int):
    tasks = client.projects.retrieve(project_id).get_tasks()
    for task_meta in tasks:
        jobs = task_meta.get_jobs()
        job_id = jobs[0].id  # we assume there is only one job per task
        job_metadata = client.jobs.retrieve(job_id).get_meta()
        frames = job_metadata.frames
        height, width = frames[0].height, frames[0].width
        anno_table = task_meta.get_annotations()
        obj_arr, label_arr = get_obj_arr_and_labels(
            anno_table, len(frames), height, width
        )
        selector = parse_selector(task_meta.name)
        yield selector, obj_arr, label_arr


# creates one-to-one mapping of nuclei to soma, based on maximum overlap
def colocalize_rois(nuc_rois, soma_rois):
    for nuc_id in np.unique(nuc_rois[np.nonzero(nuc_rois)]):
        nuc_mask = nuc_rois == nuc_id
        soma_id_contents = soma_rois[nuc_mask][np.nonzero(soma_rois[nuc_mask])]
        # sometimes there are no soma corresponding to the nuclear mask
        if soma_id_contents.size == 0:
            continue
        soma_id = np.argmax(np.bincount(soma_id_contents))
        yield (nuc_id, soma_id)


def measure_nuc_cyto_ratio_legacy(  # noqa: C901
    client: Client,
    project_id: int,
    intensity: xr.DataArray,
    nuc_channel: str,
    soma_channel: str,
    measurement_channels: list[str] | None = None,
):
    if measurement_channels is None:
        measurement_channels = intensity["channel"].values.tolist()

    df = pd.DataFrame()
    for selector, obj_arr, _ in enumerate_rois(client, project_id):
        subarr = intensity.sel(selector)
        channels = selector["channel"].tolist()

        nuc_idx = channels.index(nuc_channel)
        soma_idx = channels.index(soma_channel)

        soma_mask = obj_arr[soma_idx]
        nuclear_mask = obj_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8)

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        soma_measurements = []
        for props in regionprops(soma_mask):
            soma_measurements.append(
                {
                    "id": props.label,
                    "area_soma": props.area,
                }
            )
        soma_df = pd.DataFrame.from_records(soma_measurements)

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append(
                {
                    "id": props.label,
                    "area_cyto": props.area,
                }
            )
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append(
                {
                    "id": props.label,
                    "area_nuc": props.area,
                }
            )
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:  # type: ignore
            # sometimes these collections are inhomogenous and don't contain all the channels we're interested in
            if channel not in subarr["channel"].values:
                continue

            field_intensity_arr = subarr.sel(channel=channel).values

            assert (
                cyto_mask.shape == field_intensity_arr.shape
            ), f"cyto mask and intensity array have different shapes: {cyto_mask.shape} | {field_intensity_arr.shape}"

            # measure soma
            for props in regionprops(soma_mask, intensity_image=field_intensity_arr):
                mask = soma_mask == props.label
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_mean_soma"
                ] = field_intensity_arr[mask].mean()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_std_soma"
                ] = field_intensity_arr[mask].std()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_median_soma"
                ] = np.median(field_intensity_arr[mask])

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                mask = cyto_mask == props.label
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_mean_cyto"
                ] = field_intensity_arr[mask].mean()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_std_cyto"
                ] = field_intensity_arr[mask].std()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_median_cyto"
                ] = np.median(field_intensity_arr[mask])

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                mask = nuclear_mask == props.label
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_mean_nuc"
                ] = field_intensity_arr[mask].mean()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_std_nuc"
                ] = field_intensity_arr[mask].std()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_median_nuc"
                ] = np.median(field_intensity_arr[mask])

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id", suffixes=("_nuc", "_cyto"))
        merged.insert(0, "field", selector["field"])
        merged.insert(0, "region", selector["region"])
        df = pd.concat((df, merged))

    return df


@click.command("nuc-cyto-legacy")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.argument("nuc_channel", type=str)
@click.argument("soma_channel", type=str)
@click.option(
    "--projection",
    type=click.Choice(["none", "max", "sum"]),
    default="none",
    help="z-projection mode",
)
@click.option(
    "--channels",
    type=str,
    default="",
    help="comma-separated list of channels to measure from; defaults to all",
)
def cli_entry(
    project_name: str,
    experiment_dir: pl.Path,
    experiment_type: ExperimentType,
    nuc_channel: str,
    soma_channel: str,
    channels: str,
    projection: str,
):
    logger.info("Reading experiment directory...")
    experiment = load_experiment(experiment_dir, experiment_type)

    logger.info("Caching experiment as zarray... this may take a few minutes.")
    cache_dir = config.scratch_dir / (experiment_dir.name + ".zarr")
    atexit.register(lambda: shutil.rmtree(cache_dir, ignore_errors=True))
    ds = xr.Dataset({"intensity": experiment})
    ds.to_zarr(cache_dir, mode="w")

    intensity = xr.open_zarr(cache_dir).intensity.astype(np.float32)

    match projection:
        case "none":
            pass
        case "max":
            intensity = intensity.max("z", skipna=True)
        case "sum":
            intensity = intensity.sum("z", skipna=True)

    client = new_client_from_config(config)
    if (project := get_project(client, project_name)) is None:
        raise ValueError(f"Project {project_name} not found!")

    project_id = project.id

    if channels == "":
        channel_list = None
    else:
        channel_list = channels.split(",")

    output_dir = experiment_dir / "results"
    output_dir.mkdir(exist_ok=True)

    df = measure_nuc_cyto_ratio_legacy(
        client, project_id, intensity, nuc_channel, soma_channel, channel_list
    )
    df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)
