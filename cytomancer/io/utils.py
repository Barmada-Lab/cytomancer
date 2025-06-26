import logging
import shutil
from pathlib import Path

import xarray as xr
from acquisition_io import ExperimentType, load_experiment

from cytomancer.config import config

logger = logging.getLogger(__name__)


def stage_experiment(
    experiment_path: Path,
    experiment_type: ExperimentType,
    scratch_dir: Path | None = None,
) -> xr.Dataset:
    """Idempotently stage an experiment as a zarr for processing. Returns an
        xarray dataset backed by the staged zarr.

    Args:
        experiment_path (Path): Path to the experiment.
        experiment_type (ExperimentType): Type of experiment.
        scratch_dir (Path, optional): Path to the scratch directory. Defaults to config.scratch_dir.

    Returns:
        xr.Dataset: Xarray backed by the staged zarr.
    """
    scratch_dir = scratch_dir or config.scratch_dir
    acquisition_data = experiment_path / "acquisition_data"
    experiment = xr.Dataset(
        {"intensity": load_experiment(acquisition_data, experiment_type)}
    )

    experiment_staging = scratch_dir / experiment_path.name
    experiment_staging.mkdir(parents=True, exist_ok=True)
    zarr_path = experiment_staging / "acquisition_data.zarr"

    if zarr_path.exists():
        logger.info(f"Staged experiment already exists at {zarr_path}")
        try:
            return xr.open_zarr(zarr_path)
        except Exception as e:
            logger.error(f"Error opening staged experiment: {e}")
            shutil.rmtree(zarr_path)

    try:
        logger.info(f"Staging experiment to {zarr_path}")
        experiment.to_zarr(zarr_path, mode="w")
    except Exception as e:
        logger.error(f"Error staging experiment: {e}")
        shutil.rmtree(zarr_path)
        raise e

    return xr.open_zarr(zarr_path)
