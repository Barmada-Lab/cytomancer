import pathlib as pl

from cvat_sdk import Client, Config
from skimage.measure import regionprops
import pandas as pd
import xarray as xr
import click

from cytomancer.config import config
from cytomancer.experiment import ExperimentType, Axes
from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from .upload import prep_experiment
from .helpers import enumerate_rois


def measure_2d(
        client: Client,
        project_id: int,
        collections: dict[str, xr.DataArray],
        measurement_channels: list[str]):

    df = pd.DataFrame()
    for selector, obj_arr, _ in enumerate_rois(client, project_id):
        collection = list(collections.values())[0]
        intensity_arr = collection.sel(selector)
        for rois in obj_arr:

            field_measurements = []
            for props in regionprops(rois):
                field_measurements.append({
                    "id": props.label,
                    "region": selector[Axes.REGION],
                    "field": selector[Axes.FIELD],
                    "area": props.area,
                })
            field_df = pd.DataFrame.from_records(field_measurements)

            for channel in measurement_channels:
                field_intensity_arr = intensity_arr.sel(channel=channel).values
                print(field_intensity_arr.min(), field_intensity_arr.max())
                for props in regionprops(rois, intensity_image=field_intensity_arr):
                    mask = rois == props.label
                    field_df.loc[
                        field_df["id"] == props.label, f"{channel}_intensity_sum"] = field_intensity_arr[mask].sum()
                    field_df.loc[
                        field_df["id"] == props.label, f"{channel}_intensity_std"] = field_intensity_arr[mask].std()

            df = pd.concat((df, field_df))

    return df


@click.command("measure")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.option("--channels", type=str, default="", help="comma-separated list of channels to measure")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
@click.option("--dims", type=click.Choice(["XY", "TXY", "CXY", "ZXY"]), default="XY", help="dims of uploaded stacks")
def cli_entry(
        project_name: str,
        experiment_dir: pl.Path,
        experiment_type: ExperimentType,
        channels: str,
        mip: bool,
        dims: str):

    channel_list = channels.split(",")
    if channel_list == [""]:
        raise ValueError("Must provide at least one channel to measure")

    if experiment_type == "nd2s":
        collections = {nd2_file.name: prep_experiment(nd2_file, mip, False, experiment_type, 0.0, None, False) for nd2_file in experiment_dir.glob("**/*.nd2")}
    else:
        collections = {experiment_dir.name: prep_experiment(experiment_dir, mip, False, experiment_type, rescale=0.0, channels=None, apply_psuedocolor=False, to_uint8=False, fillna=False)}

    output_dir = experiment_dir / "results"
    output_dir.mkdir(exist_ok=True)

    client = Client(url=config.cvat_url, config=Config(verify_ssl=False))
    client.login((config.cvat_username, config.cvat_password))
    org_slug = config.cvat_org
    client.organization_slug = org_slug

    (data, _) = client.api_client.projects_api.list(search=project_name)
    assert data is not None and len(data.results) > 0, \
        f"No project matching {project_name} in {org_slug}; create a project in the webapp first"

    project = next(filter(lambda x: x.name == project_name, data.results))
    project_id = project.id

    match dims:
        case "XY":
            df = measure_2d(client, project_id, collections, channel_list)
            df.to_csv(output_dir / "measurements_CVAT.csv", index=False)
            return
        case "TXY":
            raise NotImplementedError("TXY measurements not implemented")
        case "CXY":
            raise NotImplementedError("CXY measurements not implemented")
        case "ZXY":
            raise NotImplementedError("ZXY measurements not implemented")
        case _:
            raise ValueError(f"Unknown dims {dims}")
