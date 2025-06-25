import logging
from pathlib import Path

import click
import joblib
from acquisition_io import ExperimentType

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.config import config
from cytomancer.dask import dask_client

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
    "--snr-thresh",
    type=float,
    default=2,
    help="Minimum DAPI signal-to-noise ratio to include in training data.",
)
def pultra_survival(
    experiment_dir: Path,
    classifier_name,
    save_annotations: bool,
    snr_thresh: float,
):
    """
    Run pultra survival analysis on an experiment. Note that only CQ1 acquisitions are supported.
    """
    svm_path = config.models_dir / classifier_name

    from cytomancer.models.pultra_survival import run

    with dask_client():
        run(
            experiment_dir,
            ExperimentType.CQ1,
            svm_path,
            save_annotations,
            snr_thresh,
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
    from cytomancer.models.pultra_classifier import do_train

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


@click.command("stardist-seg")
@experiment_dir_argument()
@click.option(
    "--model-name",
    type=str,
    default="2D_versatile_fluo",
    show_default=True,
    help="Name of pretrained StarDist model to use.",
)
@click.option(
    "--preprocess-method",
    type=click.Choice(["punctate", "none"]),
    default="punctate",
    show_default=True,
    help="Name of preprocessing method to use.",
)
@click.option(
    "--clahe-clip",
    type=float,
    default=0.01,
    show_default=True,
    help="CLAHE clip limit. Used for image normalization.",
)
def stardist_nuc_seg(
    experiment_dir: Path, model_name: str, preprocess_method: str, clahe_clip: float
):
    from cytomancer.models.stardist_seg import run

    with dask_client():
        run(experiment_dir, model_name, preprocess_method, clahe_clip)


@click.command(name="unet-train")
@click.option("--dataset-path", type=Path, help="Path to the training dataset")
@click.option(
    "--checkpoint-path",
    type=Path,
    default=Path("./checkpoints/"),
    help="Path to training checkpoints",
)
@click.option("--epochs", type=int, default=100, help="Number of epochs")
@click.option("--batch-size", type=int, default=16, help="Batch size")
@click.option("--learning-rate", type=float, default=1e-5, help="Learning rate")
@click.option("--momentum", type=float, default=0.999, help="Momentum")
@click.option("--amp", type=bool, default=False, help="AMP")
@click.option("--bilinear", type=bool, default=True, help="Bilinear")
@click.option("--n-classes", type=int, default=1)
@click.option("--load", type=str, default=None, help="Load model")
def unet_train(
    dataset_path: Path,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    amp: bool,
    bilinear: bool,
    n_classes: int,
    load: str | None,
):
    from cytomancer.models.unet.train import run

    run(
        dataset_path,
        checkpoint_path,
        epochs,
        batch_size,
        learning_rate,
        momentum,
        amp,
        bilinear,
        n_classes,
        load,
    )


def register(cli: click.Group):
    @cli.group("quant", help="Tools for quantifying data")
    @click.pass_context
    def quant_group(ctx):
        ctx.ensure_object(dict)

    quant_group.add_command(train_pultra_classifier)
    quant_group.add_command(pultra_survival)
    quant_group.add_command(stardist_nuc_seg)
    quant_group.add_command(unet_train)
