import logging
from pathlib import Path

from cvat_sdk import Client

from .helpers import get_project

logger = logging.getLogger(__name__)


def backup(cvat_client: Client, project_name: str, experiment_dir: Path):
    """
    exports a backup of an annotated project to an experiment's results directory
    """

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if (project := get_project(cvat_client, project_name)) is None:
        raise ValueError(f"No project with name {project_name} found")

    zip_output = results_dir / "cvat_backup.zip"
    project.download_backup(zip_output)
