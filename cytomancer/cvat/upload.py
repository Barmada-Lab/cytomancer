from pathlib import Path
import logging
import tempfile
import atexit
import shutil
import uuid

from cvat_sdk.models import TaskWriteRequest
from distributed import Client, as_completed
from tqdm import tqdm
import xarray as xr
import pandas as pd
import click
import dask
import tifffile

from cytomancer.experiment import ExperimentType
from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.utils import load_experiment, iter_idx_prod
from cytomancer.ops.display import apply_psuedocolor, clahe, rescale_intensity
from cytomancer.config import config
from .helpers import new_client_from_config, get_project, create_project, exponential_backoff


logger = logging.getLogger(__name__)


@dask.delayed
def stage_task(arr: xr.DataArray, tmpdir: Path):
    region_name = arr.coords["region"].values
    task_name = f"{region_name}-{uuid.uuid4()}"
    coords, files = [], []
    for frame in iter_idx_prod(arr, subarr_dims=["y", "x"]):
        tmpfile = tmpdir / f"{uuid.uuid4()}.tif"
        tifffile.imwrite(tmpfile, frame.data)
        coords.append(frame.coords)
        files.append(tmpfile)

    return task_name, coords, files


@exponential_backoff(max_retries=5, base_delay=0.1)
def upload_task(cvat_client, project_id, task_name, files):
    cvat_client.tasks.create_from_data(
        spec=TaskWriteRequest(
            name=task_name,
            project_id=project_id),  # type: ignore
        resources=files,
        data_params=dict(
            image_quality=100,
            sorting_method="predefined"))


@click.command("upload-experiment")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--channels", type=str, default="", help="comma-separated list of channels to include. Defaults to all channels")
@click.option("--regions", type=str, default="", help="comma-separated list of regions to include. Defaults to all regions")
@click.option("--tps", type=str, default="", help="comma-separated list of timepoints to upload. Defaults to all timepoints")
@click.option("--samples-per-region", type=int, default=-1, help="number of fields to upload per region")
@click.option("--composite", is_flag=True, default=False, help="composite channels if set, else uploads each channel separately")
@click.option("--projection", type=click.Choice(["none", "sum", "maximum_intensity"]), default="none", help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["yx", "tyx", "cyx", "zyx"]), default="yx", help="dims of uploaded stacks")
@click.option("--clahe-clip", type=float, default=0.00,
              help="""Clip limit for contrast limited adaptive histogram equalization. Enhances
              contrast for easier annotation of dim structures, but may misrepresent relative
              intensities within each field. Set above 0 to enable. """)
def cli_entry(  # noqa: C901
        project_name: str,
        experiment_dir: Path,
        experiment_type: ExperimentType,
        channels: str,
        regions: str,
        tps: str,
        samples_per_region: int,
        composite: bool,
        projection: str,
        dims: str,
        clahe_clip: float):

    # dims are provided as individual characters, need to map to xarray dims
    dim_mapping = {"t": "time", "c": "channel", "z": "z", "y": "y", "x": "x"}
    subarr_dims = [dim_mapping[dim] for dim in dims]

    experiment = load_experiment(experiment_dir, experiment_type)

    if channels:
        experiment = experiment.sel(channel=channels.split(","))

    if regions:
        experiment = experiment.sel(region=regions.split(","))

    if tps:
        experiment = experiment.isel(time=list(map(int, tps.split(","))))

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
        experiment = psuedocolor.mean("channel")
        subarr_dims.append("rgb")
    else:
        experiment = rescale_intensity(experiment, dims=["y", "x"], out_range="uint8")

    cvat_client = new_client_from_config(config)

    if (project := get_project(cvat_client, project_name)) is None:
        project = create_project(cvat_client, project_name)

    project_id = project.id

    dask_client = Client(n_workers=12, threads_per_worker=3)

    tmpdir = Path(tempfile.mkdtemp(dir=config.scratch_dir))
    atexit.register(lambda: shutil.rmtree(tmpdir))

    upload_tasks = []
    for arr in iter_idx_prod(experiment, subarr_dims=subarr_dims):
        upload_tasks.append(stage_task(arr, tmpdir))

    uploaded_coords = []
    uploaded_paths = []
    result_iter = as_completed(dask_client.compute(upload_tasks), with_results=True)
    for _, (task_name, coords, files) in tqdm(result_iter, total=len(upload_tasks)):  # type: ignore
        try:
            upload_task(cvat_client, project_id, task_name, files)
            uploaded_coords += coords
            uploaded_paths += files
        except Exception as e:
            logger.error(f"Failed to upload {task_name}: {e}. Skipping.")

    records = []
    for coord, path in zip(uploaded_coords, uploaded_paths):
        record = {dim: coord[dim].values for dim in coord}
        record["path"] = path.name
        records.append(record)

    results_dir = experiment_dir / "results"
    pd.DataFrame \
        .from_records(records) \
        .to_csv(results_dir / "cvat_upload_records.csv", index=False)
