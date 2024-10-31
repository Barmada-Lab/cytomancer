import atexit
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tifffile
import xarray as xr
from cvat_sdk.models import TaskWriteRequest

from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.ops.display import apply_psuedocolor, clahe, rescale_intensity
from cytomancer.utils import iter_idx_prod, load_experiment

from .helpers import (
    create_project,
    exponential_backoff,
    get_project,
    new_client_from_config,
)

logger = logging.getLogger(__name__)


@dataclass
class StagedTask:
    name: str
    coords: list[xr.DataArray]
    files: list[Path]


def stage_task(idx: int, arr: xr.DataArray, tmpdir: Path, blind: bool):
    region_name = arr.coords["region"].values
    task_name = f"{idx:06d}" if blind else f"{region_name}_{idx:08d}"
    coords, files = [], []
    subarr_dims = ["y", "x", "rgb"] if "rgb" in arr.dims else ["y", "x"]
    frames = list(iter_idx_prod(arr, subarr_dims=subarr_dims))
    n_pad = len(str(len(frames)))
    for idx, frame in enumerate(frames):
        padded_idx = str(idx).zfill(n_pad)
        tmpfile = tmpdir / f"{task_name}_{padded_idx}.tif"
        tifffile.imwrite(tmpfile, frame.data)
        coords.append(frame.coords)
        files.append(tmpfile)

    return StagedTask(task_name, coords, files)


def handle_staging(
    arr: xr.DataArray, tmpdir: Path, subarr_dims: list[str], blind: bool
):
    for idx, subarr in enumerate(
        iter_idx_prod(arr, subarr_dims=subarr_dims, shuffle=blind)
    ):
        yield stage_task(idx, subarr, tmpdir, blind)  # type: ignore


@exponential_backoff(max_retries=5, base_delay=0.1)
def upload_task(cvat_client, project_id, staged_task: StagedTask):
    cvat_client.tasks.create_from_data(
        spec=TaskWriteRequest(name=staged_task.name, project_id=project_id),  # type: ignore
        resources=staged_task.files,
        data_params={"image_quality": 100},
    )


def fmt_records(staged_task: StagedTask):
    for frame, coord in zip(staged_task.files, staged_task.coords, strict=False):
        yield {"frame": frame.name, **{dim: coord[dim].values for dim in coord}}


def do_upload(  # noqa: C901
    project_name: str,
    experiment: xr.DataArray,
    channels: list[str],
    regions: list[str],
    fields: list[str],
    tps: list[str],
    composite: bool,
    projection: str,
    subarr_dims: list[str],
    clahe_clip: float,
    blind: bool,
):
    logger.info("finished caching experiment.")

    if len(channels) > 0:
        experiment = experiment.sel(channel=channels)

    if len(regions) > 0:
        experiment = experiment.sel(region=regions)

    if len(fields) > 0:
        experiment = experiment.sel(field=fields)

    if len(tps) > 0:
        experiment = experiment.isel(time=[int(x) - 1 for x in tps])

    match projection:
        case "sum":
            experiment = experiment.sum("z")
        case "maximum_intensity":
            experiment = experiment.max("z")
        case "none":
            pass

    if clahe_clip > 0:
        experiment = clahe(experiment, clahe_clip)

    if composite:
        psuedocolor = apply_psuedocolor(experiment)
        experiment = psuedocolor.mean("channel").astype("uint8")
        subarr_dims.append("rgb")
    else:
        experiment = rescale_intensity(experiment, dims=["y", "x"], out_range="uint8")

    cvat_client = new_client_from_config(config)

    # TODO: we can probably implement idempotency here to handle cases where uploads are interrupted
    if (project := get_project(cvat_client, project_name)) is not None:
        raise ValueError(
            f"Project {project_name} taken! Please choose a different name or delete the pre-existing project."
        )

    project = create_project(cvat_client, project_name)

    project_id = project.id

    logger.info("staging tasks")

    logger.info(experiment)
    records = []
    with tempfile.TemporaryDirectory(dir=config.scratch_dir) as tmpdir:
        for staged in handle_staging(experiment, Path(tmpdir), subarr_dims, blind):
            upload_task(cvat_client, project_id, staged)
            records += fmt_records(staged)

        logger.info("finished staging")

    return pd.DataFrame.from_records(records)


def upload_experiment(
    experiment_dir: Path,
    experiment_type: ExperimentType,
    project_name: str,
    channels: str,
    regions: str,
    fields: str,
    tps: str,
    composite: bool,
    projection: str,
    dims: str,
    clahe_clip: float,
    blind: bool,
):
    if project_name == "":
        project_name = experiment_dir.name

    logger.info("Reading experiment directory...")
    experiment = load_experiment(experiment_dir, experiment_type)

    logger.info("Caching experiment as zarray... this may take a few minutes.")
    cache_dir = config.scratch_dir / (experiment_dir.name + ".zarr")
    atexit.register(lambda: shutil.rmtree(cache_dir, ignore_errors=True))
    ds = xr.Dataset({"intensity": experiment})
    ds.to_zarr(cache_dir, mode="w")

    experiment = xr.open_zarr(cache_dir).intensity

    channels_list = channels.split(",") if channels else []
    fields_list = fields.split(",") if fields else []
    regions_list = regions.split(",") if regions else []
    tps_list = tps.split(",") if tps else []

    # dims are provided as individual characters, need to map to xarray dims
    dim_mapping = {"t": "time", "c": "channel", "z": "z", "y": "y", "x": "x"}
    subarr_dims = [dim_mapping[dim] for dim in dims]

    df = do_upload(
        project_name=project_name,
        experiment=experiment,
        channels=channels_list,
        regions=regions_list,
        fields=fields_list,
        tps=tps_list,
        composite=composite,
        projection=projection,
        subarr_dims=subarr_dims,
        clahe_clip=clahe_clip,
        blind=blind,
    )

    cvat_upload_records = experiment_dir / "results" / "cvat_upload.csv"
    cvat_upload_records.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cvat_upload_records, index=False)
