import logging
from pathlib import Path

import click
import joblib

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.config import config
from cytomancer.dask import dask_client
from cytomancer.experiment import ExperimentType

logger = logging.getLogger(__name__)


@click.command("pult-surv")
@experiment_dir_argument()
@click.option(
    "--classifier-name",
    default="nuclei_survival_svm.joblib",
    show_default=True,
    help="Name of pretrained StarDist model to use.",
)
@click.option(
    "--save-annotations", is_flag=True, help="Save annotated stacks to results folder"
)
@click.option(
    "--run-sync",
    is_flag=True,
    help="Run the task synchronously, skipping the task queue.",
)
@click.option(
    "--snr-thresh",
    type=float,
    default=2,
    help="Minimum DAPI signal-to-noise ratio to include in training data.",
)
def pultra_survival(
    experiment_dir: Path,
    classifier_name,
    save_annotations: bool,
    run_sync: bool,
    snr_thresh: float,
):
    """
    Run pultra survival analysis on an experiment. Note that only CQ1 acquisitions are supported.
    """
    svm_path = config.models_dir / classifier_name

    if run_sync:
        from cytomancer.quant.pultra_survival import run

        with dask_client():
            run(
                experiment_dir,
                ExperimentType.CQ1,
                svm_path,
                save_annotations,
                snr_thresh,
            )
    else:
        from cytomancer.quant.tasks import run_pultra_survival

        run_pultra_survival.delay(
            str(experiment_dir), ExperimentType.CQ1, str(svm_path), save_annotations
        )


@click.command("train-pultra-classifier")
@click.argument("cvat_project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.argument(
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
)
@click.argument("live_label", type=str)
@click.option(
    "--min-dapi-snr",
    type=float,
    default=2,
    help="Minimum DAPI signal-to-noise ratio to include in training data.",
)
@click.option(
    "--dump-predictions", is_flag=True, help="Dump predictions to disk for debugging."
)
def train_pultra_classifier(
    cvat_project_name,
    experiment_dir: Path,
    experiment_type: ExperimentType,
    output_path: Path,
    live_label: str,
    min_dapi_snr: float,
    dump_predictions: bool,
):
    """
    Train a classifier for pultra survival analysis.
    """
    from cytomancer.quant.pultra_classifier import do_train

    classifier = do_train(
        cvat_project_name,
        experiment_dir,
        experiment_type,
        live_label,
        min_dapi_snr,
        dump_predictions,
    )
    if classifier is not None:
        joblib.dump(classifier, output_path)
        logger.info(f"Saved classifier to {output_path}")


@click.command("neurite-quant")
@experiment_dir_argument()
@experiment_type_argument()
@click.option(
    "--model-name",
    type=str,
    default="ilastish_neurite_seg.joblib",
    help="Path to ilastik model",
)
def neurite_quant(
    experiment_dir: Path, experiment_type: ExperimentType, model_name: str
):
    model_path = config.models_dir / model_name

    from cytomancer.quant.tasks import run_neurite_quant

    run_neurite_quant.delay(str(experiment_dir), experiment_type, str(model_path))


@click.command("stardist-nuc-seg")
@experiment_dir_argument()
def stardist_nuc_seg(experiment_dir: Path):
    from cytomancer.quant.stardist_nuc_seg import run

    with dask_client():
        run(experiment_dir)


def register(cli: click.Group):
    @cli.group("quant", help="Tools for quantifying data")
    @click.pass_context
    def quant_group(ctx):
        ctx.ensure_object(dict)

    quant_group.add_command(train_pultra_classifier)
    quant_group.add_command(pultra_survival)
    quant_group.add_command(stardist_nuc_seg)
    # quant_group.add_command(neurite_quant)
