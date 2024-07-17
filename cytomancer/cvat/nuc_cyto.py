import pathlib as pl

from cvat_sdk import Client, Config
from skimage.measure import regionprops
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import click

from cytomancer.config import config
from cytomancer.experiment import ExperimentType
from cytomancer.click_utils import experiment_dir_argument, experiment_type_argument
from .upload import prep_experiment
from .helpers import enumerate_rois


# creates one-to-one mapping of nuclei to soma, based on maximum overlap
def colocalize_rois(nuc_rois, soma_rois):
    for nuc_id in np.unique(nuc_rois[np.nonzero(nuc_rois)]):
        nuc_mask = nuc_rois == nuc_id
        soma_id_contents = soma_rois[nuc_mask][np.nonzero(soma_rois[nuc_mask])]
        # sometimes there are no soma corresponding to the nuclear mask
        if soma_id_contents.size == 0:
            continue
        soma_id = np.argmax(np.bincount(soma_id_contents))
        yield (nuc_id, soma_id)


# temp hack until we figure out how to work with inhomogenous experiments
def measure_nuc_cyto_ratio_nd2s(  # noqa: C901
        client: Client,
        project_id: int,
        collection_name: str,
        intensity_arr: xr.DataArray,
        nuc_channel: str,
        soma_channel: str,
        measurement_channels: list[str] | None = None):

    if measurement_channels is None:
        measurement_channels = intensity_arr["channel"].values.tolist()

    df = pd.DataFrame()
    for selector, obj_arr, _ in enumerate_rois(client, project_id):

        region = selector.pop("region")
        if region != collection_name:
            continue

        intensity_arr = intensity_arr.sel(selector)

        channels = selector["channel"]
        nuc_idx = np.where(channels == nuc_channel)
        soma_idx = np.where(channels == soma_channel)

        soma_mask = obj_arr[soma_idx]
        nuclear_mask = obj_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8)

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        soma_measurements = []
        for props in regionprops(soma_mask):
            soma_measurements.append({
                "id": props.label,
                "area_soma": props.area,
            })
        soma_df = pd.DataFrame.from_records(soma_measurements)

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append({
                "id": props.label,
                "area_cyto": props.area,
            })
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append({
                "id": props.label,
                "area_nuc": props.area,
            })
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:  # type: ignore

            print(f"Measuring {channel}")
            # sometimes these collections are inhomogenous and don't contain all the channels we're interested in
            if channel not in intensity_arr["channel"].values:
                continue

            field_intensity_arr = intensity_arr.sel(channel=channel).values

            # measure soma
            for props in regionprops(soma_mask, intensity_image=field_intensity_arr):
                mask = soma_mask == props.label
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_mean_soma"] = field_intensity_arr[mask].mean()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_std_soma"] = field_intensity_arr[mask].std()

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                mask = cyto_mask == props.label
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_mean_cyto"] = field_intensity_arr[mask].mean()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_std_cyto"] = field_intensity_arr[mask].std()

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                mask = nuclear_mask == props.label
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_mean_nuc"] = field_intensity_arr[mask].mean()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_std_nuc"] = field_intensity_arr[mask].std()

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id").merge(soma_df, on="id")
        merged.insert(0, "field", selector["field"])
        merged.insert(0, "region", region)
        df = pd.concat((df, merged))

    return df


def measure_nuc_cyto_ratio(  # noqa: C901
        client: Client,
        project_id: int,
        collection: xr.DataArray,
        nuc_channel: str,
        soma_channel: str,
        measurement_channels: list[str] | None = None):

    if measurement_channels is None:
        measurement_channels = collection["channel"].values.tolist()

    df = pd.DataFrame()
    for selector, obj_arr, _ in enumerate_rois(client, project_id):

        intensity_arr = collection.sel(selector)
        channels = selector["channel"].tolist()

        nuc_idx = channels.index(nuc_channel)
        soma_idx = channels.index(soma_channel)

        soma_mask = obj_arr[soma_idx]
        nuclear_mask = obj_arr[nuc_idx]
        cyto_mask = soma_mask * (~nuclear_mask.astype(bool)).astype(np.uint8)

        if cyto_mask.max() == 0 or nuclear_mask.max() == 0:
            continue

        soma_measurements = []
        for props in regionprops(soma_mask):
            soma_measurements.append({
                "id": props.label,
                "area_soma": props.area,
            })
        soma_df = pd.DataFrame.from_records(soma_measurements)

        cytoplasmic_measurements = []
        for props in regionprops(cyto_mask):
            cytoplasmic_measurements.append({
                "id": props.label,
                "area_cyto": props.area,
            })
        cyto_df = pd.DataFrame.from_records(cytoplasmic_measurements)

        nuclear_measurements = []
        for props in regionprops(nuclear_mask):
            nuclear_measurements.append({
                "id": props.label,
                "area_nuc": props.area,
            })
        nuc_df = pd.DataFrame.from_records(nuclear_measurements)

        for channel in measurement_channels:  # type: ignore
            # sometimes these collections are inhomogenous and don't contain all the channels we're interested in
            if channel not in intensity_arr["channel"].values:
                continue

            field_intensity_arr = intensity_arr.sel(channel=channel).values

            assert cyto_mask.shape == field_intensity_arr.shape, \
                f"cyto mask and intensity array have different shapes: {cyto_mask.shape} | {field_intensity_arr.shape}"

            # measure soma
            for props in regionprops(soma_mask, intensity_image=field_intensity_arr):
                mask = soma_mask == props.label
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_mean_soma"] = field_intensity_arr[mask].mean()
                soma_df.loc[
                    soma_df["id"] == props.label, f"{channel}_std_soma"] = field_intensity_arr[mask].std()

            # measure cyto
            for props in regionprops(cyto_mask, intensity_image=field_intensity_arr):
                mask = cyto_mask == props.label
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_mean_cyto"] = field_intensity_arr[mask].mean()
                cyto_df.loc[
                    cyto_df["id"] == props.label, f"{channel}_std_cyto"] = field_intensity_arr[mask].std()

            # measure nuc
            for props in regionprops(nuclear_mask, intensity_image=field_intensity_arr):
                mask = nuclear_mask == props.label
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_mean_nuc"] = field_intensity_arr[mask].mean()
                nuc_df.loc[
                    nuc_df["id"] == props.label, f"{channel}_std_nuc"] = field_intensity_arr[mask].std()

        colocalized = dict(colocalize_rois(nuclear_mask, soma_mask))
        nuc_df["id"] = nuc_df["id"].map(colocalized)
        merged = nuc_df.merge(cyto_df, on="id", suffixes=("_nuc", "_cyto"))
        merged.insert(0, "field", selector["field"])
        merged.insert(0, "region", selector["region"])
        df = pd.concat((df, merged))

    return df


@click.command("nuc-cyto")
@click.argument("project_name", type=str)
@experiment_dir_argument()
@experiment_type_argument()
@click.argument("nuc_channel", type=str)
@click.argument("soma_channel", type=str)
@click.option("--channels", type=str, default="", help="comma-separated list of channels to measure from; defaults to all")
@click.option("--mip", is_flag=True, default=False, help="apply MIP to each z-stack")
def cli_entry(
        project_name: str,
        experiment_dir: pl.Path,
        experiment_type: ExperimentType,
        nuc_channel: str,
        soma_channel: str,
        channels: str,
        mip: bool,
        ):

    client = Client(url=config.cvat_url, config=Config(verify_ssl=False))
    client.login((config.cvat_username, config.cvat_password))
    org_slug = config.cvat_org
    client.organization_slug = org_slug
    api_client = client.api_client

    (data, _) = api_client.projects_api.list(search=project_name)
    assert data is not None and len(data.results) > 0, \
        f"No project matching {project_name} in organization {org_slug}"

    try:
        # exact matches only
        project = next(filter(lambda x: x.name == project_name, data.results))
    except StopIteration:
        raise ValueError(f"No project matching {project_name} in organization {org_slug}")

    if channels == "":
        channel_list = None
    else:
        channel_list = channels.split(",")

    project_id = project.id
    output_dir = experiment_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # TODO: homogenize collections and put into one array
    if experiment_type is ExperimentType.ND2:
        df = pd.DataFrame()
        for nd2_file in tqdm(experiment_dir.glob("**/*.nd2")):
            collection_name = nd2_file.name.replace(".nd2", "")
            intensity_arr = prep_experiment(nd2_file, mip, False, experiment_type, 0.0, None, False, False, False)
            print(intensity_arr)
            collection_df = measure_nuc_cyto_ratio_nd2s(client, project_id, collection_name, intensity_arr, nuc_channel, soma_channel, channel_list)
            df = pd.concat((df, collection_df), ignore_index=True)
            intensity_arr.close()
        df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)
    else:
        collections = prep_experiment(experiment_dir, mip, False, experiment_type, 0.0, None, False, False, False)
        df = measure_nuc_cyto_ratio(client, project_id, collections, nuc_channel, soma_channel, channel_list)
        df.to_csv(output_dir / "nuc_cyto_CVAT.csv", index=False)
