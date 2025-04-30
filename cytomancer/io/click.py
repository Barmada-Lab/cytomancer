from pathlib import Path

import click

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.utils import load_experiment

from .dump import dump_tiffs_cli


@click.command("stage-experiment")
@experiment_dir_argument()
@experiment_type_argument()
@click.option(
    "-s",
    "--scratch-dir",
    default=config.scratch_dir,
    type=click.Path(exists=True, file_okay=False),
    help="Path to scratch directory",
)
def stage_experiment(
    experiment_dir: Path, experiment_type: ExperimentType, scratch_dir: Path
):
    experiment_staging = scratch_dir / experiment_dir.name
    experiment_staging.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(experiment_dir, experiment_type)
    experiment.to_zarr(experiment_staging / "acquisition_data.zarr", mode="w")


def register(cli: click.Group):
    @cli.group("io", help="I/O utilities")
    @click.pass_context
    def io_group(ctx):
        ctx.ensure_object(dict)

    io_group.add_command(dump_tiffs_cli)
