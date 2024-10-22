import pathlib as pl
from typing import Any

import click
import matplotlib.pyplot as plt
import pandas as pd
from cvat_sdk import Client, Config
from lifelines import CoxPHFitter
from tqdm import tqdm

from cytomancer.config import config

from .nuc_cyto_legacy import parse_selector


def extract_survival_result(track, length) -> dict[str, Any]:
    points = track.shapes
    for point in points:
        if point.outside:
            return {"time": point.frame, "dead": 1}
    return {"time": length - 1, "dead": 0}


def analyze_survival(
    client: Client, project_id: int, output_dir: pl.Path, well_csv: pl.Path | None
):
    tasks = client.projects.retrieve(project_id).get_tasks()
    rows = []
    for task_meta in tqdm(tasks):
        length = task_meta.size
        try:
            well = parse_selector(task_meta.name)["region"]
        except ValueError:
            well = task_meta.name[:3]
        annotation = task_meta.get_annotations()
        for track in annotation.tracks:
            survival_result = extract_survival_result(track, length)
            survival_result["well"] = well
            rows.append(survival_result)
    df = pd.DataFrame.from_records(rows)

    if well_csv is not None:
        well_df = pd.read_csv(well_csv)[["Vertex", "Condition"]]
        well_df["well"] = well_df["Vertex"].str[-3:]
        well_df = well_df.drop(columns=["Vertex"])
        df = df.merge(well_df, on="well")
        condition_counts = df.groupby("Condition").size()
        df["Condition"] += (
            " (n=" + df["Condition"].map(condition_counts).astype(str) + ")"
        )
        cph = CoxPHFitter()
        print(df)
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
        output_fig = output_dir / "CoxPH_baselines_CVAT.pdf"
        plt.savefig(output_fig, format="pdf")

    output_csv = output_dir / "survival_CVAT.csv"
    df.to_csv(output_csv, index=False)


@click.command("legacy-survival")
@click.argument("project_name")
@click.argument("output_dir", type=click.Path(path_type=pl.Path))
@click.option("--well-csv", type=click.Path(path_type=pl.Path), default=None)
def cli_entry(project_name: str, output_dir: pl.Path, well_csv: pl.Path | None):
    client = Client(url=config.cvat_url, config=Config(verify_ssl=False))
    client.login((config.cvat_username, config.cvat_password))
    org_slug = config.cvat_org
    client.organization_slug = org_slug
    api_client = client.api_client

    (data, _) = api_client.projects_api.list(search=project_name)
    assert (
        data is not None and len(data.results) > 0
    ), f"No project matching {project_name} in organization {org_slug}"

    try:
        # exact matches only
        project = next(filter(lambda x: x.name == project_name, data.results))
    except StopIteration:
        raise ValueError(
            f"No project matching {project_name} in organization {org_slug}"
        )

    output_dir.mkdir(exist_ok=True)

    project_id = project.id
    analyze_survival(client, project_id, output_dir, well_csv)
