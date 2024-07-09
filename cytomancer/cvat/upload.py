import warnings
import time
from typing import Callable
import pathlib as pl
import tempfile
import random
import logging

from cvat_sdk import Client as CvatClient
from cvat_sdk import Config
from cvat_sdk.models import TaskWriteRequest, ProjectWriteRequest
from skimage.exposure import rescale_intensity
from toolz import curry
from dask.distributed import Client as DaskClient
import xarray as xr
import tifffile
import click

from cytomancer.ops import display
from cytomancer.config import config
from cytomancer.experiment import Axes, ExperimentType
from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.utils import load_experiment
from .helpers import coord_selector

logger = logging.getLogger(__name__)


@curry
def stage_single_frame(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    selector_label = coord_selector(arr)
    outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
    tifffile.imwrite(outpath, arr)
    return [outpath]


@curry
def stage_t_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for t in arr[Axes.TIME]:
        frame = arr.sel({Axes.TIME: t})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images


@curry
def stage_channel_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for c in arr[Axes.CHANNEL]:
        frame = arr.sel({Axes.CHANNEL: c})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images


@curry
def stage_z_stack(arr: xr.DataArray, tmpdir: pl.Path) -> list[pl.Path]:
    images = []
    for z in arr[Axes.Z]:
        frame = arr.sel({Axes.Z: z})
        selector_label = coord_selector(frame)
        outpath = pl.Path(tmpdir) / f"{selector_label}.tif"
        tifffile.imwrite(outpath, frame)
        images.append(outpath)
    return images


@curry
def stage_basic_tiff(path: pl.Path, tmpdir: pl.Path) -> list[pl.Path]:
    img = tifffile.imread(path)
    rescaled = rescale_intensity(img, out_range="uint8")
    outpath = tmpdir / path.name
    tifffile.imwrite(outpath, rescaled)
    return [outpath]


def upload(client, project_id: int, label: str, images: list[pl.Path]):
    for i in range(5):
        try:
            client.tasks.create_from_data(
                spec=TaskWriteRequest(
                    name=label,
                    project_id=project_id),
                resources=images,
                data_params=dict(
                    image_quality=100,
                    sorting_method="predefined"))
            return
        except Exception as e:
            if i == 4:
                logger.error(f"Failed to upload {label} after 5 attempts. Skipping.")
            else:
                logger.warn(f"Error uploading {label}: {e}")
                logger.warn(f"Retrying in {i ** 2} seconds")
                time.sleep(i ** 2)


def stage_and_upload(
        client,
        project_id: int,
        label: str,
        stage_arr: Callable[[pl.Path], list[pl.Path]]):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pl.Path(tmpdir)
        images = stage_arr(tmpdir)
        upload(client, project_id, label, images)


def prep_experiment(
        experiment_base: pl.Path,
        mip: bool,
        composite: bool,
        experiment_type: ExperimentType,
        rescale: float,
        channels: str | list[str] | None,
        apply_psuedocolor: bool = True,
        to_uint8: bool = True,
        fillna: bool = True):

    intensity = load_experiment(experiment_base, experiment_type, fillna)

    attrs = intensity.attrs

    if channels is not None:
        intensity = intensity.sel({Axes.CHANNEL: channels})

    if mip:
        if Axes.Z not in intensity.dims:
            raise ValueError("MIP requested but no z-dimension found")
        intensity = intensity.max(dim=Axes.Z)

    if apply_psuedocolor:
        intensity = display.apply_psuedocolor(intensity).assign_attrs(attrs)

    if composite:
        if Axes.CHANNEL not in intensity.dims:
            warnings.warn("Composite requested but no channel dimension found; ignoring")
        intensity = intensity.mean(dim=Axes.CHANNEL)

    if to_uint8:
        intensity = display.rescale_intensity(
            intensity, [Axes.Y, Axes.X], in_percentile=(rescale, 100 - rescale), out_range="uint8")

    return intensity


@click.command("upload-raw")
@click.argument("project_name", type=str)
@click.argument("collection_base", type=click.Path(exists=True, file_okay=False, path_type=pl.Path))
def cli_entry_basic(
        project_name: str,
        collection_base: pl.Path):

    client = CvatClient(url=config.cvat_url, config=Config(verify_ssl=False))
    client.login((config.cvat_username, config.cvat_password))

    org_slug = config.cvat_org
    client.organization_slug = org_slug

    (data, _) = client.api_client.projects_api.list(search=project_name)
    if data is None or len(data.results) == 0:
        (project, _) = client.api_client.projects_api.create(
            ProjectWriteRequest(name=project_name)  # type: ignore
        )

    else:
        project = next(filter(lambda x: x.name == project_name, data.results))

    project_id = project.id  # type: ignore

    for path in collection_base.glob("**/*.tif"):
        stage_and_upload(client, project_id, path.stem, stage_basic_tiff(path))  # type: ignore


@click.command("upload-experiment")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--channels", type=str, default="", help="comma-separated list of channels to include. Defaults to all channels")
@click.option("--regions", type=str, default="", help="comma-separated list of regions to include. Defaults to all regions")
@click.option("--tps", type=str, default="", help="comma-separated list of timepoints to upload. Defaults to all timepoints")
@click.option("--composite", is_flag=True, default=False, help="composite channels if set, else uploads each channel separately")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["XY", "TXY", "CXY", "ZXY"]), default="XY", help="dims of uploaded stacks")
@click.option("--rescale", type=float, default=0.0,
              help="""rescales images by stretching the range of their values to be bounded
                by the given percentile range, e.g. a value of 1 will rescale an image
                so that 0 1st percentile and 255 is the 99th percentile""")
@click.option("--samples-per-region", type=int, default=-1, help="number of fields to upload per region")
@click.option("--fillna", is_flag=True, default=False, help="interpolate missing images")
def cli_entry_experiment(
        project_name: str,
        experiment_base: pl.Path,
        channels: str,
        regions: str,
        tps: str,
        composite: bool,
        mip: bool,
        dims: str,
        experiment_type: ExperimentType,
        rescale: float,
        samples_per_region: int,
        fillna: bool):

    dask_client = DaskClient()
    logger.info(f"Dashboard link: {dask_client.dashboard_link}")

    channel_list = None if channels == "" else channels.split(",")
    if channel_list is not None and len(channel_list) == 1:
        channel_list = channel_list[0]

    if experiment_type == ExperimentType.ND2:
        logger.info("Loading ND2 files... this may take a while.")
        collections = [prep_experiment(nd2_file, mip, composite, experiment_type, rescale, channel_list, apply_psuedocolor=True, fillna=fillna) for nd2_file in experiment_base.glob("**/*.nd2")]  # noqa: E501
    else:
        collections = [prep_experiment(experiment_base, mip, composite, experiment_type, rescale, channel_list, apply_psuedocolor=True, fillna=fillna)]  # noqa: E501

    client = CvatClient(url=config.cvat_url)
    client.login((config.cvat_username, config.cvat_password))

    org_slug = config.cvat_org
    client.organization_slug = org_slug

    (data, _) = client.api_client.projects_api.list(search=project_name)
    if data is None or len(data.results) == 0:
        (project, _) = client.api_client.projects_api.create(
            ProjectWriteRequest(name=project_name)  # type: ignore
        )

    else:
        project = next(filter(lambda x: x.name == project_name, data.results))

    project_id = project.id  # type: ignore

    for collection in collections:
        logger.info(f"uploading {collection.coords[Axes.REGION]}")
        if tps != "":
            tps_list = [int(tp) for tp in tps.split(",")]
            if len(tps_list) > 1:
                collection = collection.isel({Axes.TIME: tps_list})
            else:
                collection = collection.isel({Axes.TIME: tps_list[0]})
        if regions != "":
            regions_list = [region for region in regions.split(",")]
            collection = collection.sel({Axes.REGION: regions_list})
        match dims:
            case "XY":
                assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                for region in collection[Axes.REGION]:
                    region_arr = collection.sel({Axes.REGION: region})
                    sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)  # noqa: E501
                    for field in sample:
                        arr = region_arr.sel({Axes.FIELD: field})
                        selector_label = coord_selector(arr)
                        stage_and_upload(client, project_id, selector_label, stage_single_frame(arr))  # type: ignore

            case "TXY":
                assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.TIME, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                for region in collection[Axes.REGION]:
                    sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)  # noqa: E501
                    region_arr = collection.sel({Axes.REGION: region})
                    for field in sample:
                        arr = region_arr.sel({Axes.FIELD: field})
                        selector_label = coord_selector(arr)
                        stage_and_upload(client, project_id, selector_label, stage_t_stack(arr))  # type: ignore

            case "CXY":
                assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.CHANNEL, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                for region in collection[Axes.REGION]:
                    region_arr = collection.sel({Axes.REGION: region})
                    sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                    for field in sample:
                        arr = region_arr.sel({Axes.FIELD: field})
                        selector_label = coord_selector(arr)
                        stage_and_upload(client, project_id, selector_label, stage_channel_stack(arr))  # type: ignore

            case "ZXY":
                assert {*collection.dims} == {Axes.REGION, Axes.FIELD, Axes.Z, Axes.X, Axes.Y, Axes.RGB}, collection.dims
                for region in collection[Axes.REGION]:
                    region_arr = collection.sel({Axes.REGION: region})
                    sample = collection[Axes.FIELD] if samples_per_region == -1 else random.sample([field for field in collection[Axes.FIELD]], samples_per_region)
                    for field in sample:
                        arr = region_arr.sel({Axes.FIELD: field})
                        selector_label = coord_selector(arr)
                        stage_and_upload(client, project_id, selector_label, stage_z_stack(arr))  # type: ignore

            case _:
                raise ValueError(f"Unknown dims {dims}")

    logger.info("Done!")
