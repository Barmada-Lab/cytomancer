from pathlib import Path

import pandas as pd
from cvat_sdk.models import LabeledShapeRequest, ShapeType, TaskAnnotationsUpdateRequest
from skimage import exposure, filters, morphology  # type: ignore
from stardist.models import StarDist2D

from cytomancer.config import config
from cytomancer.cvat.helpers import (
    get_project,
    get_project_label_map,
    get_rles,
    new_client_from_config,
)
from cytomancer.experiment import ExperimentType
from cytomancer.utils import load_experiment


def run(
    project_name: str,
    experiment_dir: Path,
    experiment_type: ExperimentType,
    channel: str,
    label_name: str,
    adapteq_clip_limit: float,
    median_filter_d: int,
    model_name: str,
):
    client = new_client_from_config(config)
    if (project := get_project(client, project_name)) is None:
        print(f"No projects matching query '{project_name}' found.")
        return

    tasks = project.get_tasks()

    if len(tasks) == 0:
        print(f"No tasks found in project '{project_name}'.")
        return

    project_id = project.id
    label_map = get_project_label_map(client, project_id)

    if label_name not in label_map:
        print(f"No labels matching query '{label_name}' found.")
        print(
            f"Please create a label with the name '{label_name}' in project '{project_name}'."
        )
        return

    label_id = label_map[label_name]

    upload_record_location = experiment_dir / "results" / "cvat_upload.csv"
    if not upload_record_location.exists():
        print(f"No upload record found at {upload_record_location}.")
        return

    dtype_spec = {"channel": str, "z": str, "region": str, "field": str}
    task_df = pd.read_csv(
        upload_record_location, dtype=dtype_spec, parse_dates=["time"]
    ).set_index("frame")

    intensity = load_experiment(experiment_dir, experiment_type)
    model = StarDist2D.from_pretrained(model_name)

    for task in tasks:
        job = task.get_jobs()[0]
        selectors = [
            task_df.loc[frame.name].to_dict() for frame in job.get_frames_info()
        ]
        channels = [selector["channel"] for selector in selectors]

        chan_idx = channels.index(channel)
        selector = selectors[chan_idx]
        selector["time"] = pd.Timestamp(selector["time"], unit="ns")
        frame = intensity.sel(selector).values
        eqd = exposure.equalize_adapthist(frame, clip_limit=adapteq_clip_limit)
        med = filters.median(eqd, morphology.disk(median_filter_d))
        preds, _ = model.predict_instances(med)  # type: ignore

        shapes = []
        for _id, rle in get_rles(preds):  # type: ignore
            shapes.append(
                LabeledShapeRequest(
                    type=ShapeType("mask"),
                    points=rle,
                    label_id=label_id,
                    frame=chan_idx,
                )
            )

        client.api_client.tasks_api.update_annotations(
            id=task.id,
            task_annotations_update_request=TaskAnnotationsUpdateRequest(shapes=shapes),
        )
