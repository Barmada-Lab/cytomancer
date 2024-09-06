from typing import Any
import pathlib as pl

from tqdm import tqdm
import pandas as pd
import click

from cytomancer.cvat.helpers import new_client_from_config, get_project
from cytomancer.click_utils import experiment_dir_argument
from cytomancer.config import config


def extract_survival_result(track, length) -> dict[str, Any]:
    points = track.shapes
    for point in points:
        if point.outside:
            return {"time": point.frame, "dead": 1}
    return {"time": length - 1, "dead": 0}


def analyze_survival(tasks) -> pd.DataFrame:

    rows = []
    for task_meta in tqdm(tasks):
        length = task_meta.size
        task_name = task_meta.name
        annotation = task_meta.get_annotations()
        for track in annotation.tracks:
            survival_result = extract_survival_result(track, length)
            survival_result["task_name"] = task_name
            rows.append(survival_result)

    return pd.DataFrame.from_records(rows)


@click.command("survival")
@click.argument("project_name")
@experiment_dir_argument()
def cli_entry(project_name: str, experiment_dir: pl.Path):

    client = new_client_from_config(config)

    if (project := get_project(client, project_name)) is None:
        raise ValueError(f"No project matching {project_name}")

    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    survival_output = results_dir / "cvat_survival.csv"

    analyze_survival(project.tasks).to_csv(survival_output, index=False)
