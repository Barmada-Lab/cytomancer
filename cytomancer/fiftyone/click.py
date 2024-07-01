from pathlib import Path
import shutil
import time

from dask.distributed import Client
import fiftyone as fo
import click

from cytomancer.click_utils import experiment_dir_argument
from cytomancer.dask import dask_client


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


@click.command("ingest", help="Ingest a CQ1 dataset into FiftyOne. Other formats are not yet supported.")
@experiment_dir_argument()
def ingest(experiment_dir: Path) -> None:
    with dask_client() as _:
        from .ingest import ingest_cq1_longitudinal as ingest_fiftyone
        ingest_fiftyone(experiment_dir)


def register(cli: click.Group) -> None:
    @cli.group("fiftyone")
    def fiftyone() -> None:
        pass

    fiftyone.add_command(ingest)
    fiftyone.add_command(launch_app)
    fiftyone.add_command(delete_dataset)
