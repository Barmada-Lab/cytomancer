import pathlib as pl

import dask.array as da
import xarray as xr

from . import ioutils


def load_legacy(base: pl.Path, fillna: bool) -> xr.DataArray:
    timepoint_tags = sorted(
        {int(path.name.replace("T", "")) for path in base.glob("raw_imgs/*/*")}
    )
    region_tags = set()
    field_tags = set()
    channel_tags = set()
    for path in base.glob("raw_imgs/**/*.tif"):
        channel_tags.add(path.parent.parent.parent.name)
        region, field = path.name.split(".")[0].split("_")
        region_tags.add(region)
        field_tags.add(field)  # get rid of zero-padding

    region_tags = sorted(region_tags)
    channel_tags = sorted(channel_tags)
    field_tags = sorted(field_tags, key=lambda x: int(x))
    timepoint_tags = sorted(timepoint_tags)

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT

    channels = []
    for channel in channel_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            regions = []
            for region in region_tags:
                fields = []
                for field in field_tags:
                    col = region[1:]
                    path = (
                        base
                        / "raw_imgs"
                        / channel
                        / f"T{timepoint}"
                        / f"col_{col}"
                        / f"{region}_{field}.tif"
                    )
                    img = ioutils.read_tiff_toarray(path)
                    fields.append(img)
                regions.append(da.stack(fields))
            timepoints.append(da.stack(regions))
        channels.append(da.stack(timepoints))
    plate = da.stack(channels)

    intensity = xr.DataArray(
        plate,
        dims=["channel", "time", "region", "field", "y", "x"],
        coords={
            "channel": channel_tags,
            "time": timepoint_tags,
            "region": region_tags,
            "field": [str(int(field)) for field in field_tags],
        },
    )

    if fillna:
        intensity = intensity.ffill("time").bfill("time").ffill("field").bfill("field")

    return intensity


def load_legacy_icc(base: pl.Path, fillna: bool) -> xr.DataArray:
    timepoint_tags = sorted(
        {int(path.name.replace("T", "")) for path in base.glob("raw_imgs/*/*")}
    )
    region_tags = set()
    field_tags = set()
    channel_tags = set()
    for path in base.glob("raw_imgs/**/*.tif"):
        channel_tags.add(path.parent.parent.parent.name)
        region, field = path.name.split(".")[0].split("_")
        region_tags.add(region)
        field_tags.add(field)  # get rid of zero-padding

    region_tags = sorted(region_tags)
    channel_tags = sorted(channel_tags)
    field_tags = sorted(field_tags, key=lambda x: int(x))
    timepoint_tags = sorted(timepoint_tags)

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT

    channels = []
    for channel in channel_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            regions = []
            for region in region_tags:
                fields = []
                for field in field_tags:
                    col = region[1:]
                    path = (
                        base
                        / "raw_imgs"
                        / channel
                        / f"T{timepoint}"
                        / f"col_{col}"
                        / f"{region}_{field}.tif"
                    )
                    img = ioutils.read_tiff_toarray(path)
                    fields.append(img)
                regions.append(da.stack(fields))
            timepoints.append(da.stack(regions))
        channels.append(da.stack(timepoints))
    plate = da.stack(channels)

    intensity = xr.DataArray(
        plate,
        dims=["channel", "region", "time", "field", "y", "x"],
        coords={
            "channel": channel_tags,
            "time": [0],
            "region": list(map(str, timepoint_tags)),
            "field": [str(int(field)) for field in field_tags],
        },
    ).squeeze("time", drop=True)

    if fillna:
        intensity = intensity.ffill("time").bfill("time").ffill("field").bfill("field")

    return intensity
