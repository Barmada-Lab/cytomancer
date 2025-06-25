import logging
from pathlib import Path

from cvat_sdk import Client

logger = logging.getLogger(__name__)


def restore(cvat_client: Client, experiment_dir: Path):
    """
    restores a project from a canonical backup location in the experiment's
    results directory
    """

    backup_zip = experiment_dir / "results" / "cvat_backup.zip"
    cvat_client.projects.create_from_backup(backup_zip)
