from pathlib import Path

import click

from cytomancer.experiment import ExperimentType
from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument


@click.command("stardist-seg-cvat-proj")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--channel", "-c", default="DAPI", show_default=True, help="Channel to segment.")
@click.option("--label-name", "-l", default="dead", show_default=True, help="Name of the label to create.")
@click.option("--adapteq-clip-limit", "-c", default=0.01, show_default=True, help="Clip limit for adaptive histogram equalization.")
@click.option("--median-filter-d", "-m", default=5, show_default=True, help="Diameter of the median filter to apply to the images before segmentation.")
@click.option("--model-name", "-m", default="2D_versatile_fluo", show_default=True, help="Name of predefined StarDist model to use.")
def stardist_seg_cvat_proj(
        project_name: str,
        experiment_dir: Path,
        experiment_type: ExperimentType,
        channel: str,
        label_name: str,
        adapteq_clip_limit: float,
        median_filter_d: int,
        model_name: str):

    """ Segment images in a CVAT project using StarDist. """
    from cytomancer.oneoffs.stardist_seg_cvat_proj import run
    run(
        project_name,
        experiment_dir,
        experiment_type,
        channel,
        label_name,
        adapteq_clip_limit,
        median_filter_d,
        model_name)


def register(cli: click.Group):
    @cli.group("oneoffs", help="A place for scripts that I wrote once and might be useful again")
    @click.pass_context
    def oneoffs_group(ctx):
        ctx.ensure_object(dict)

    oneoffs_group.add_command(stardist_seg_cvat_proj)
