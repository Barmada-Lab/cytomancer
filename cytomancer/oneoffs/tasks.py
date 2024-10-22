from pathlib import Path

from cytomancer.celery import CytomancerTask, app
from cytomancer.experiment import ExperimentType


@app.task(bind=True)
def stardist_seg_cvat_proj_run(
    self: CytomancerTask,  # noqa: ARG001
    project_name: str,
    experiment_dir: str,
    experiment_type: ExperimentType,
    channel: str,
    label_name: str,
    adapteq_clip_limit: float,
    median_filter_d: int,
    model_name: str,
):
    from .stardist_seg_cvat_proj import run

    run(
        project_name,
        Path(experiment_dir),
        experiment_type,
        channel,
        label_name,
        adapteq_clip_limit,
        median_filter_d,
        model_name,
    )
