from pathlib import Path
import shutil
import inspect
import time

import fiftyone as fo
import click

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.dask import dask_client
from cytomancer.experiment import ExperimentType
from .import_survival import do_import_survival_results
from .ingest import do_ingest_cq1
from .zhuzh import zhuzh
from .import_survival import do_import_survival


@click.command("launch-app")
def launch_app() -> None:
    fo.launch_app(address="0.0.0.0")  # type: ignore
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down fiftyone... please wait...")
            fo.close_app()
            print("Done.")
            return


@click.command("delete-dataset")
@click.argument("dataset_name")
def delete_dataset(dataset_name: str) -> None:
    dataset = fo.load_dataset(dataset_name)
    shutil.rmtree(dataset.info["media_dir"])  # type: ignore
    fo.delete_dataset(dataset_name)


@click.command("ingest")
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--name", default="", help="Name of the dataset to create; defaults to the name of experiment_dir")
@click.option("--regions", default="", help="Comma-separated list of regions to ingest")
@click.option("--fields", default="", help="Comma-separated list of fields to ingest")
@click.option("--channels", default="", help="Comma-separated list of channels to ingest")
@click.option("--timepoints", default="", help="Comma-separated list of timepoints to ingest")
def ingest(experiment_dir: Path, experiment_type: ExperimentType, name: str, regions, fields, channels, timepoints) -> None:
    regions = regions.split(",") if regions else []
    fields = fields.split(",") if fields else []
    timepoints = [int(t) for t in timepoints.split(",")] if timepoints else []
    channels = channels.split(",") if channels else []
    name = name if name else experiment_dir.name
    with dask_client():
        do_ingest_cq1(experiment_dir, name, regions, fields, channels, timepoints)


@click.command("import-survival")
@experiment_dir_argument()
@click.option("--name", default="", help="Name of the importing fiftyone dataset; defaults to the name of experiment_dir")
def import_survival(experiment_dir: Path, name: str) -> None:
    name = name if name else experiment_dir.name
    do_import_survival(experiment_dir, name)


@click.command("zhuzh", help=inspect.getdoc(zhuzh))
@click.argument("dataset_name")
def run_zhuzh(dataset_name: str) -> None:
    with dask_client():
        zhuzh(dataset_name)


def register(cli: click.Group) -> None:
    @cli.group("fiftyone")
    def fiftyone() -> None:
        pass

    fiftyone.add_command(ingest)
    fiftyone.add_command(launch_app)
    fiftyone.add_command(delete_dataset)
    fiftyone.add_command(run_zhuzh)
    fiftyone.add_command(import_survival)
