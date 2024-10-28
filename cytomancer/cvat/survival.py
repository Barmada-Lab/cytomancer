import pathlib as pl
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm

from cytomancer.click_utils import experiment_dir_argument
from cytomancer.config import config
from cytomancer.cvat.helpers import get_project, new_client_from_config


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
@click.option(
    "--well-csv",
    type=click.Path(exists=True, path_type=pl.Path),
    help="CSV file with well information",
)
@click.option(
    "--ignore-missing-upload", is_flag=True, help="Ignore missing upload record"
)
def cli_entry(
    project_name: str,
    experiment_dir: pl.Path,
    well_csv: pl.Path | None,
    ignore_missing_upload: bool,
):
    client = new_client_from_config(config)

    if (project := get_project(client, project_name)) is None:
        raise ValueError(f"No project matching {project_name}")

    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    survival_output = results_dir / "cvat_survival.csv"

    upload_record_path = results_dir / "cvat_upload.csv"
    if (upload_record_path).exists():
        task_df = pd.read_csv(upload_record_path).drop(columns=["time"])
        task_df["frame"] = task_df["frame"].str.replace(r"_\d+\.tif", "", regex=True)
        task_df.rename(columns={"frame": "task_name"}, inplace=True)
        task_df.drop_duplicates(inplace=True)
        df = (
            analyze_survival(project.get_tasks())
            .merge(task_df, on="task_name")
            .rename(columns={"region": "well"})
            .drop(columns=["task_name", "field"])
            .sort_values(["time", "dead", "well"])
        )
        if well_csv is not None:
            well_df = pd.read_csv(well_csv)[["Vertex", "Condition"]]
            well_df["well"] = well_df["Vertex"].str[-3:]
            well_df = well_df.drop(columns=["Vertex"])
            df = df.merge(well_df, on="well")
            df.to_csv(survival_output, index=False)
            condition_counts = df.groupby("Condition").size()
            df["Condition"] += (
                " (n=" + df["Condition"].map(condition_counts).astype(str) + ")"
            )
            cph = CoxPHFitter()
            cph.fit(
                df.drop(columns="well"),
                duration_col="time",
                event_col="dead",
                strata="Condition",
            )
            cph.baseline_cumulative_hazard_.plot(
                ylabel="Cumulative hazard",
                xlabel="T",
                title="Baseline cumulative hazards",
                drawstyle="steps-mid",
            )
            output_fig = results_dir / "CoxPH_baselines_CVAT.pdf"
            plt.savefig(output_fig, format="pdf")

    elif ignore_missing_upload:
        df = analyze_survival(project.get_tasks()).sort_values(
            ["task_name", "time", "dead"]
        )
        df.to_csv(survival_output, index=False)

    else:
        raise FileNotFoundError(
            f"Could not find {upload_record_path}; was this project uploaded with an older version of cytomancer? If so, use the legacy survival script"
        )
