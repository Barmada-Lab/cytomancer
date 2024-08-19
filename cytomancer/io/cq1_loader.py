from typing import Tuple
import pathlib as pl
from itertools import product
import xml.etree.ElementTree as xml
import warnings
import logging

from datetime import datetime
import tifffile
import dask.array as da
import dask
import pandas as pd
import ome_types
import xarray as xr
import numpy as np
import re


logger = logging.getLogger(__name__)

CQ1_ACQUISITION_DIR_REGEX = r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T(?P<hr>\d{2})(?P<min>\d{2})(?P<sec>\d{2})_(?P<plate_name>.*)$"
CQ1_WELLPLATE_NAME_REGEX = r"^W(?P<well_idx>\d*)\(.*\),A.*,F(?P<field_idx>\d*)$"

CHANNEL_EX_EM_LUT = {
    (405, 447): "DAPI",
    (488, 525): "GFP",
    (561, 617): "RFP"
}

PLATE_WELL_LUT = {
    (8, 12): [r+c for r, c in product(["A", "B", "C", "D", "E", "F", "G", "H"], ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])],
}


def _try_parse_dir(path: pl.Path) -> datetime | None:
    if (match := re.match(CQ1_ACQUISITION_DIR_REGEX, path.name)) is not None:
        return datetime(
            year=int(match["year"]),
            month=int(match["month"]),
            day=int(match["day"]),
            hour=int(match["hr"]),
            minute=int(match["min"]),
            second=int(match["sec"]),
        )
    return None


def get_tp_df(path: pl.Path, ome_xml_filename: str):  # noqa: C901, get bent flake8

    ome_xml = ome_types.from_xml(path / ome_xml_filename)

    result_xml_path = path / "ImagingResult.xml"
    assert result_xml_path.exists(), f"Could not find ImagingResult.xml in {path}."
    result_xml = xml.parse(result_xml_path)

    info = result_xml.find("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}ResultInfo")
    start_time_str = info.get("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}BeginTime")  # type: ignore
    end_time_str = info.get("{http://www.yokogawa.co.jp/LSC/ICMSchema/1.0}EndTime")  # type: ignore
    fmt = "%Y-%m-%dT%H:%M:%S"
    start_time = datetime.strptime(start_time_str.split(".")[0], fmt)  # type: ignore
    end_time = datetime.strptime(end_time_str.split(".")[0], fmt)  # type: ignore
    acquisition_delta = end_time - start_time

    plate = ome_xml.plates[0]
    rows, cols = plate.rows, plate.columns
    if (rows, cols) in PLATE_WELL_LUT:
        wells = PLATE_WELL_LUT[(rows, cols)]  # type: ignore
    else:
        warnings.warn("Could not find well names for this plate size. Falling back to integer-based well names. Consider adding plate size to PLATE_WELL_LUT in cq1_loader.py.")
        wells = [f"{d:02}_{c:02}" for d, c in product(range(rows), range(cols))]  # type: ignore

    ex_px = ome_xml.images[1].pixels

    channels = []
    for channel in ex_px.channels:
        if channel.illumination_type.value == "Epifluorescence":  # type: ignore
            ex, em = int(channel.excitation_wavelength), int(channel.emission_wavelength)  # type: ignore
            channel_str = CHANNEL_EX_EM_LUT.get((ex, em), f"{ex}nm/{em}nm")  # type: ignore
            channels.append(channel_str)
        else:
            channels.append(channel.contrast_method.value)  # type: ignore

    if len(channels) != len(set(channels)):
        # fallback to integer-based channel names
        channels = list(range(len(channels)))

    shape = (ex_px.size_x, ex_px.size_y)
    attrs = dict(
        ome_xml_filename=ome_xml_filename,
        px_size_x=ex_px.physical_size_x,
        px_size_y=ex_px.physical_size_y,
        px_size_z=ex_px.physical_size_z)  # note: for MIPs, means the z step size of the original images

    records = []
    for image in ome_xml.images[1:]:  # skip the first image, which contains only metadata
        assert image.name is not None, f"Image {image.id} has no name."
        image_match = re.match(CQ1_WELLPLATE_NAME_REGEX, image.name)
        assert image_match is not None, f"Failed to parse image name: {image.name}"

        well_idx = int(image_match["well_idx"]) - 1
        well_label = wells[well_idx]
        field_idx_label = image_match["field_idx"]

        pixels = image.pixels

        for plane, data in zip(pixels.planes, pixels.tiff_data_blocks):
            z = plane.the_z
            t = plane.the_t
            c = channels[plane.the_c]
            assert z is not None and t is not None and c is not None, f"Image plane is missing coordinate information: {plane}."
            assert data.uuid is not None, f"Data block {data.id} has no UUID."
            image_path = data.uuid.file_name
            assert image_path is not None, f"Data block {data.id} has no file name."
            records.append({
                "time": t,
                "channel": c,
                "region": well_label,
                "field": field_idx_label,
                "z": z,
                "path": path / image_path,
            })

    df = pd.DataFrame.from_records(records)
    df = df[["time", "channel", "region", "field", "z", "path"]]  # explicitly order columns
    ts = df["time"].unique().size
    acq_delta = acquisition_delta / ts
    df["time"] = df["time"].map(lambda t: start_time + acq_delta * t).astype("datetime64[ns]")

    preliminary_mi = pd.MultiIndex.from_frame(df.drop(["path"], axis=1))
    holy_mi = pd.MultiIndex.from_product(preliminary_mi.levels, names=preliminary_mi.names)
    holy_df = df[["path"]].set_index(preliminary_mi).reindex(index=holy_mi).sort_index().replace({np.nan: None})

    return holy_df, shape, attrs


def get_experiment_df_detailed(base_path: pl.Path, measurement_type: str = "mip", ordinal_time: bool = False) -> Tuple[pd.DataFrame, tuple, dict]:
    """
    Indexes a CQ1 experiment directory by timepoint, channel, region, field, and z-slice.

    Parameters
    ----------
    base_path : pl.Path
        Path to the directory containing the experiment.

    ordinal_time : bool
        If True, reindex the timepoints to be ordinal integers. This is useful for displaying
        longitudinal experiments in a more human-readable format. only applies to multi-timepoint
        experiments.

    Returns
    -------
    pd.DataFrame
    """

    def reindex_time(df, value):
        multiindex = df.index
        time_index = multiindex.names.index("time")
        new_index = multiindex.set_levels(
            multiindex.levels[time_index].map(lambda _: value),
            level=time_index
        )
        return df.set_index(new_index)

    match measurement_type.strip().lower():
        case "mip":
            measurement_file = "MeasurementResultMIP.ome.xml"
        case "sum":
            measurement_file = "MeasurementResultSUM.ome.xml"
        case "raw":
            measurement_file = "MeasurementResult.ome.xml"
        case _:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    acquisitions = list(base_path.glob(f"*/{measurement_file}"))
    if len(acquisitions) == 0:
        raise ValueError("Could not find any acquisition directories in {base_path}.")

    tps = [_try_parse_dir(acq.parent) for acq in acquisitions]
    if None in tps:
        raise ValueError(f"One or more acquisition directories in {base_path} are not named according to the CQ1 convention, and cannot be parsed. \
                          Please verify acquisition directory names.")

    dt_paths = [(tp, path) for tp, path in zip(tps, acquisitions) if tp is not None]
    dt_paths = sorted(dt_paths, key=lambda d: d[0])

    shape, attrs = None, None
    df = pd.DataFrame()
    for i, (_, path) in enumerate(dt_paths):
        tp_df, tp_shape, tp_attrs = get_tp_df(path.parent, measurement_file)
        assert shape is None or shape == tp_shape, f"Shape mismatch: {shape} vs {tp_shape}"
        assert attrs is None or attrs == tp_attrs, f"Attribute mismatch: {attrs} vs {tp_attrs}"
        shape, attrs = tp_shape, tp_attrs
        if ordinal_time:
            tp_df = reindex_time(tp_df, i)
        df = pd.concat([df, tp_df])
    return df, shape, attrs  # type: ignore


def get_experiment_df(base_path: pl.Path, ordinal_time: bool = False) -> pd.DataFrame:
    return get_experiment_df_detailed(base_path, ordinal_time=ordinal_time)[0]


def load_df(df, shape, attrs) -> xr.DataArray:

    def read_img(path):
        logger.debug(f"Reading {path}")
        if path is None:
            logger.warning("MeasurementResult.ome.xml is missing an image. This is likely the result of an acquisition error! Replacing with NaNs...")
            return np.full(shape, np.nan)
        elif not path.exists():
            logger.warning(f"Could not find image at {path}, even though its existence is recorded in MeasurementResult.ome.xml. The file may have been moved or deleted. Replacing with NaNs...")
            return np.full(shape, np.nan)
        return tifffile.imread(path).astype(np.float16)

    def read_indexed_ims(recurrence):
        """
        Recursively read and stack images from a sorted hierarchical index
        in a breadth-first manner.
        """
        if type(recurrence) is pd.Series:
            path = recurrence["path"]
            return da.from_delayed(dask.delayed(read_img)(path), shape, dtype=np.float16)
        else:  # type(recurrence) is pd.DataFrame
            if type(recurrence.index) is pd.MultiIndex:
                level = recurrence.index.levels[0]  # type: ignore
            else:  # type(recurrence.index) is pd.Index
                level = recurrence.index.values
            return da.stack([read_indexed_ims(recurrence.loc[idx]) for idx in level])

    arr = read_indexed_ims(df)

    labels = df.index.names
    arr = xr.DataArray(
        arr,
        dims=labels + ["y", "x"],
        coords=dict((label, val) for label, val in zip(labels, df.index.levels)),  # type: ignore
        attrs=attrs).isel({"y": slice(0, 1998), "x": slice(0, 1998)})

    # The above method will produce coordinates of dtype object, which
    # causes issues downstream as it's inconsistent with other experiment loaders. '
    # So we explicitly cast them to strings here /shrug
    arr.coords["channel"] = arr.coords["channel"].astype(str)
    arr.coords["region"] = arr.coords["region"].astype(str)
    arr.coords["field"] = arr.coords["field"].astype(str)

    # Squeeze out Z if we're dealing with MIPs
    if "z" in arr.dims:
        arr = arr.squeeze("z", drop=True)
    return arr


def load_cq1(base_path: pl.Path) -> xr.DataArray:
    """Load a CQ1 experiment from a directory.

    Parameters
    ----------
    base_path : pl.Path
        Path to the directory containing the experiment. This can either be a directory
        containing one or more acquisition subdirectories, or an individual acquisition
        directory.

    Returns
    -------
    xr.Dataset
        The experiment data.
    """

    df, shape, attrs = get_experiment_df_detailed(base_path)
    return load_df(df, shape, attrs)
