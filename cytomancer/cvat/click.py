import click

from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from cytomancer.config import config
from cytomancer.dask import dask_client
from .nuc_cyto_legacy import cli_entry as nuc_cyto_legacy
from .measure import measure_experiment, measurement_fn_lut
from .helpers import test_cvat_credentials


@click.command("auth")
@click.option("--cvat-username", prompt="CVAT Username")
@click.password_option("--cvat-password", prompt="CVAT Password", confirmation_prompt=False)
def auth(cvat_username, cvat_password):
    """
    Update CVAT credentials. Run this with no arguments to get an interactive prompt that hides your password.
    """

    print(f"\nTesting CVAT connection to server {config.cvat_url}...")
    if not test_cvat_credentials(config.cvat_url, cvat_username, cvat_password):
        print("Connection failed. Please verify your credentials and try again.")
        print("See `cyto config update --help` for other CVAT-related settings")
        return

    print("Authentication successful. Saving credentials.")
    config.cvat_username = cvat_username
    config.cvat_password = cvat_password
    config.save()


@click.command("upload-experiment")
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--project_name", type=str, default="", help="Name of the CVAT project to create. Defaults to experiment directory name")
@click.option("--channels", type=str, default="", help="comma-separated list of channels to include. Defaults to all channels")
@click.option("--regions", type=str, default="", help="comma-separated list of regions to include. Defaults to all regions")
@click.option("--tps", type=str, default="", help="comma-separated list of timepoints to upload. Defaults to all timepoints")
@click.option("--composite", is_flag=True, default=False, help="composite channels if set, else uploads each channel separately")
@click.option("--projection", type=click.Choice(["none", "sum", "maximum_intensity"]), default="none", help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["yx", "tyx", "cyx", "zyx"]), default="yx", help="dims of uploaded stacks")
@click.option("--clahe-clip", type=float, default=0.00,
              help="""Clip limit for contrast limited adaptive histogram equalization. Enhances
              contrast for easier annotation of dim structures, but may misrepresent relative
              intensities within each field. Set above 0 to enable. """)
@click.option("--blind", is_flag=True, default=False, help="Remove identifying metadata from task names and shuffle upload order")
def upload_experiment(*args, **kwargs):
    from .upload import upload_experiment as upload_experiment_impl
    with dask_client():
        upload_experiment_impl(*args, **kwargs)


@click.command("measure")
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--roi-set-name", type=str, default="cvat_instances_default.json", help="Name of the ROI set to measure")
@click.option("--measurements", type=click.Choice(list(measurement_fn_lut.keys())), multiple=True, help="Measurements to perform on each ROI")
@click.option("--z-projection-mode", type=click.Choice(["none", "sum", "mip"]), default="none", help="Method for z-projection")
@click.option("--roi-broadcasting", type=click.Choice(["channel", "z", "time"]), multiple=True, help="Broadcasting mode for ROIs")
def measure(experiment_dir, experiment_type, roi_set_name, measurements, z_projection_mode, roi_broadcasting):
    measure_experiment(
        experiment_dir=experiment_dir,
        experiment_type=experiment_type,
        roi_set_name=roi_set_name,
        measurement_names=measurements,
        z_projection_mode=z_projection_mode,
        roi_broadcasting=roi_broadcasting)


@click.command("nuc-cyto")
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--roi-set-name", type=str, default="cvat_instances_default.json", help="Name of the ROI set to measure")
@click.option("--nuc-label", type=str, default="nucleus", help="Name of the nucleus label in the ROI set")
@click.option("--soma-label", type=str, default="soma", help="Name of the soma label in the ROI set")
def nuc_cyto(experiment_dir, experiment_type, roi_set_name, nuc_label, soma_label):
    from .colocalize import nuc_cyto_coloc_measure
    nuc_cyto_coloc_measure(experiment_dir, experiment_type, roi_set_name, nuc_label, soma_label)


def register(cli: click.Group):
    @cli.group("cvat", help="Tools for working with CVAT")
    @click.pass_context
    def cvat_group(ctx):
        ctx.ensure_object(dict)

    from cytomancer.cvat.survival import cli_entry as cvat_survival

    cvat_group.add_command(cvat_survival)

    cvat_group.add_command(auth)
    cvat_group.add_command(upload_experiment)
    cvat_group.add_command(measure)
    cvat_group.add_command(nuc_cyto)
    cvat_group.add_command(nuc_cyto_legacy)
