from pathlib import Path

import click
from acquisition_io import ExperimentType


def experiment_dir_argument(**kwargs):
    return click.argument(
        "experiment_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        **kwargs,
    )


def experiment_type_argument(**kwargs):
    return click.argument(
        "experiment_type",
        type=click.Choice(ExperimentType.__members__),  # type: ignore
        callback=lambda c, p, v: getattr(ExperimentType, v) if v else None,
        **kwargs,
    )
