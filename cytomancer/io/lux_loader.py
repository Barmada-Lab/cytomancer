import pathlib as pl

import dask.array as da
import xarray as xr

from . import ioutils


def load_lux(base: pl.Path, fillna: bool = True) -> xr.DataArray:
    timepoint_tags = sorted([int(path.name.replace("T", "")) for path in base.glob("raw_imgs/*")])
    region_tags = set()
    field_tags = set()
    exposure_tags = set()
    for path in base.glob("raw_imgs/*/*.tif"):
        region, field, exposure = path.name.split(".")[0].split("-")
        region_tags.add(region)
        field_tags.add(field)
        exposure_tags.add(exposure)

    region_tags = sorted(region_tags)
    field_tags = sorted(field_tags)
    exposure_tags = sorted(exposure_tags)
    timepoint_tags = sorted(timepoint_tags)

    # TODO: SURE SEEMS LIKE THIS COULD BE MADE RECURSIVE DONT IT
    channels = []
    for exposure in exposure_tags:
        timepoints = []
        for timepoint in timepoint_tags:
            regions = []
            for region in region_tags:
                fields = []
                for field in field_tags:
                    path = base / "raw_imgs" / f"T{timepoint}" / f"{region}-{field}-{exposure}.tif"
                    img = ioutils.read_tiff_toarray(path)
                    fields.append(img)
                regions.append(da.stack(fields))
            timepoints.append(da.stack(regions))
        channels.append(da.stack(timepoints))
    plate = da.stack(channels)

    region_coords = [region.replace("well_", "") for region in region_tags]
    field_coords = [field.replace("mosaic_", "") for field in field_tags]
    channel_coords = [exposure.split("_")[0] for exposure in exposure_tags]

    intensity = xr.DataArray(
        plate,
        dims=["channel", "time", "region", "field", "y", "x"],
        coords={
            "channel": channel_coords,
            "time": timepoint_tags,
            "region": region_coords,
            "field": field_coords,
        }
    )

    if fillna:
        intensity = intensity.ffill("time").bfill("time").ffill("field").bfill("field")

    return intensity
