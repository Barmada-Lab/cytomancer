from pathlib import Path

from cytomancer.celery import CytomancerTask, app
from cytomancer.dask import dask_client
from cytomancer.experiment import ExperimentType


@app.task(bind=True)
def run_pultra_survival(
    self: CytomancerTask,  # noqa: ARG001
    experiment_path: str,
    experiment_type: ExperimentType,
    svm_model_path: str,
    save_annotations: bool,
    snr_threshold: float,
):
    from .pultra_survival import run

    with dask_client() as _:
        run(
            Path(experiment_path),
            experiment_type,
            Path(svm_model_path),
            save_annotations,
            snr_threshold,
        )


@app.task(bind=True)
def run_neurite_quant(
    self: CytomancerTask,  # noqa: ARG001
    experiment_path: str,
    experiment_type: ExperimentType,
    ilastish_model_path: str,
):
    from .neurite_quant import run

    with dask_client() as _:
        run(Path(experiment_path), experiment_type, Path(ilastish_model_path))
