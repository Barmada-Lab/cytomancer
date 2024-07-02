from pathlib import Path

from cytomancer.celery import app, CytomancerTask
from cytomancer.experiment import ExperimentType
from cytomancer.dask import dask_client


@app.task(bind=True)
def run_pultra_survival(self: CytomancerTask, experiment_path: str, experiment_type: ExperimentType, svm_model_path: str, save_annotations: bool):
    from .pultra_survival import run
    with dask_client() as _:
        run(Path(experiment_path), experiment_type, Path(svm_model_path), save_annotations)


@app.task(bind=True)
def run_neurite_quant(self: CytomancerTask, experiment_path: str, experiment_type: ExperimentType, ilastish_model_path: str):
    from .neurite_quant import run
    with dask_client() as _:
        run(Path(experiment_path), experiment_type, Path(ilastish_model_path))
