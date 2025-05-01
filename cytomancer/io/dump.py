from pathlib import Path

import click
import numpy as np
import tifffile
from acquisition_io import ExperimentType, load_experiment
from tqdm import tqdm
from xarray import DataArray

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.ops.display import apply_psuedocolor

from ..utils import iter_idx_prod


def frame_name(frame: DataArray) -> str:
    return "-".join(f"{k}_{v.values}" for k, v in frame.coords.items())


def dump_tiffs(
    *,
    experiment_dir: Path,
    experiment_type: ExperimentType,
    output_dir: Path,
    channels: list[str],
    regions: list[str],
    fields: list[str],
    tps: list[str],
    composite: bool,
    projection: str,
    dims: list[str],
):
    experiment = load_experiment(experiment_dir, experiment_type)
    print("Loaded experiment:")
    print(experiment)

    if not output_dir.exists():
        output_dir.mkdir()

    if any(channels):
        experiment = experiment.sel(channel=channels)

    if any(regions):
        experiment = experiment.sel(region=regions)

    if any(fields):
        experiment = experiment.sel(field=fields)

    if any(tps):
        experiment = experiment.sel(tp=tps)

    if composite:
        experiment = apply_psuedocolor(experiment)
        experiment = experiment.mean("channel").astype("uint8")
        dims.append("rgb")

    match projection:
        case "sum":
            experiment = experiment.sum("z")
        case "maximum_intensity":
            experiment = experiment.max("z")
        case "none":
            pass

    for frame in tqdm(iter_idx_prod(experiment, dims)):
        name = frame_name(frame)
        if np.isnan(frame.values).any():
            continue
        tifffile.imwrite(output_dir / f"{name}.tif", frame.values)


@click.command("dump-tiffs")
@experiment_dir_argument()
@experiment_type_argument()
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--channels",
    type=str,
    default="",
    help="comma-separated list of channels to include. Defaults to all channels",
)
@click.option(
    "--regions",
    type=str,
    default="",
    help="comma-separated list of regions to include. Defaults to all regions",
)
@click.option(
    "--fields",
    type=str,
    default="",
    help="comma-separated list of fields to include. Defaults to all fields",
)
@click.option(
    "--tps",
    type=str,
    default="",
    help="comma-separated list of timepoints to include. Defaults to all timepoints",
)
@click.option(
    "--composite",
    is_flag=True,
    default=False,
    help="composite channels if set, else dumps each channel separately",
)
@click.option(
    "--projection",
    type=click.Choice(["none", "sum", "maximum_intensity"]),
    default="none",
    help="apply MIP to each z-stack",
)
@click.option(
    "--dims",
    type=click.Choice(["yx", "tyx", "cyx", "zyx"]),
    default="yx",
    help="dims of output stacks",
)
def dump_tiffs_cli(
    experiment_dir: Path,
    experiment_type: ExperimentType,
    output_dir: Path,
    channels: str,
    regions: str,
    fields: str,
    tps: str,
    composite: bool,
    projection: str,
    dims: str,
):
    channels_list = channels.split(",")
    regions_list = regions.split(",")
    fields_list = fields.split(",")
    tps_list = tps.split(",")
    dims_list = dims.split(",")

    dump_tiffs(
        experiment_dir=experiment_dir,
        experiment_type=experiment_type,
        output_dir=output_dir,
        channels=channels_list,
        regions=regions_list,
        fields=fields_list,
        tps=tps_list,
        composite=composite,
        projection=projection,
        dims=dims_list,
    )
