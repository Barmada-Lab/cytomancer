import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

from .helpers import get_project

logger = logging.getLogger(__name__)


def export(cvat_client, project_name: str, export_path: Path, format: str):
    if (project := get_project(cvat_client, project_name)) is None:
        raise ValueError(f"No project with name {project_name} found")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_output = tmp_path / "zippy.zip"
        project.export_dataset(format, zip_output, include_images=False)

        with zipfile.ZipFile(zip_output, "r") as zip_ref:
            zip_ref.extractall(tmp)

        for file in (tmp_path / "annotations").iterdir():
            export_file = export_path / f"cvat_{file.name}"
            shutil.move(file, export_file)
            logger.info(f"Exported {export_file}")


def do_export(
    cvat_client, project_name: str, experiment_dir: Path, format: str = "COCO 1.0"
):
    """
    exports annotations from cvat to an experiment's results directory
    """

    results_dir = experiment_dir / "results" / "annotations"
    results_dir.mkdir(parents=True, exist_ok=True)

    export(cvat_client, project_name, results_dir, format)
