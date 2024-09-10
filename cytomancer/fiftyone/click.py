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
def ingest(experiment_dir: Path, experiment_type: ExperimentType, name) -> None:
    with dask_client():
        do_ingest_cq1(experiment_dir)


@click.command("import-survival")
@experiment_dir_argument()
@click.option("--dataset-name", default="", help="Name of the dataset to import to; defaults to the name of experiment_dir")
def import_survival(experiment_dir: Path, dataset_name: str) -> None:
    if dataset_name == "":
        dataset_name = experiment_dir.name
    do_import_survival_results(experiment_dir, dataset_name)


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
