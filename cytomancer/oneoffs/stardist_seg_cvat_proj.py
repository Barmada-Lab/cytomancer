from pathlib import Path

from skimage import exposure, filters, morphology  # type: ignore
from stardist.models import StarDist2D
from cvat_sdk.models import TaskAnnotationsUpdateRequest, ShapeType, LabeledShapeRequest
import numpy as np

from cytomancer.cvat.helpers import new_client_from_config, parse_selector, get_project, get_project_label_map
from cytomancer.cvat.helpers import get_rles
from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.utils import load_experiment


def run(project_name: str, experiment_dir: Path, experiment_type: ExperimentType, channel: str, label_name: str, adapteq_clip_limit: float, median_filter_d: int, model_name: str):

    client = new_client_from_config(config)
    project = get_project(client, project_name)

    if project is None:
        print(f"No projects matching query '{project_name}' found.")
        return

    project_id = project.id

    tasks = project.get_tasks()

    if len(tasks) == 0:
        print(f"No tasks found in project '{project_name}'.")
        return

    label_map = get_project_label_map(client, project_id)

    if label_name not in label_map:
        print(f"No labels matching query '{label_name}' found.")
        print(f"Please create a label with the name '{label_name}' in project '{project_name}'.")
        return

    label_id = label_map[label_name]

    intensity = load_experiment(experiment_dir, experiment_type)
    model = StarDist2D.from_pretrained(model_name)

    for task in tasks:
        selector = parse_selector(task.name)

        chan_idx = int(np.where(selector["channel"] == channel)[0][0])
        selector["channel"] = np.array(channel)
        frame = intensity.sel(selector).values
        eqd = exposure.equalize_adapthist(frame, clip_limit=adapteq_clip_limit)
        med = filters.median(eqd, morphology.disk(median_filter_d))
        preds, _ = model.predict_instances(med)  # type: ignore

        shapes = []
        for id, rle in get_rles(preds):  # type: ignore
            shapes.append(
                LabeledShapeRequest(
                    type=ShapeType("mask"),
                    points=rle,
                    label_id=label_id,
                    frame=chan_idx,
                ))

        client.api_client.tasks_api.update_annotations(
            id=task.id,
            task_annotations_update_request=TaskAnnotationsUpdateRequest(
                shapes=shapes
            )
        )
