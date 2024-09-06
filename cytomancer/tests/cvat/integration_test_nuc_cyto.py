from pathlib import Path
import tempfile

from skimage import morphology
from pycocotools.coco import COCO
import pandas as pd
import xarray as xr
import numpy as np
import cv2

from cytomancer.cvat.upload import handle_upload
from cytomancer.cvat.colocalize import handle_nuc_cyto
from cytomancer.cvat.helpers import new_client_from_config, get_project
from cytomancer.cvat.export import handle_export
from cytomancer.config import config


def draw_random_circles(n_circles: int, radius: int, bimask: np.ndarray):
    for _ in range(n_circles):
        x = np.random.randint(0, bimask.shape[1])
        y = np.random.randint(0, bimask.shape[0])
        cv2.circle(bimask, (x, y), radius, 1, -1)  # type: ignore
    return bimask


def make_synthetic_nuc_cyto_data(gfp_nuc_i: int = 64, gfp_cyto_i: int = 32, dapi_nuc_i: int = 128):
    blank = np.zeros((2000, 2000), dtype="uint8")
    soma = draw_random_circles(10, 100, blank).astype(bool)
    dapi = morphology.binary_erosion(soma, morphology.disk(30)).astype("uint8") * dapi_nuc_i
    gfp = soma.astype("uint8") * gfp_cyto_i
    gfp[np.where(dapi > 0)] = gfp_nuc_i
    images = np.stack([gfp, dapi], axis=0).reshape((1, 1, 1, 2, 2000, 2000))
    coords = {
        "time": ["1"],
        "region": ["B02"],
        "field": ["1"],
        "channel": ["GFP", "DAPI"]
        }
    return xr.DataArray(images, dims=["time", "region", "field", "channel", "y", "x"], coords=coords)


def upload_synthetic_nuc_cyto_data(output_dir: Path):
    experiment = make_synthetic_nuc_cyto_data()
    task_df = handle_upload("synthetic_nuc_cyto", experiment, ["GFP", "DAPI"], [], [], False, "none", ["channel", "x", "y"], 0, False)
    zarr_output = output_dir / "synthetic_nuc_cyto.zarr"
    ds = xr.Dataset(dict(intensity=experiment))
    ds.to_zarr(zarr_output, mode="w")
    task_output = output_dir / "synthetic_nuc_cyto_task.csv"
    task_df.to_csv(task_output, index=False)


def measure_synthetic_nuc_cyto_data(output_dir: Path):
    experiment = xr.open_zarr(output_dir / "synthetic_nuc_cyto.zarr").intensity
    dtype_spec = {"channel": str, "z": str, "region": str, "field": str}
    task_df = pd.read_csv(output_dir / "synthetic_nuc_cyto_task.csv", dtype=dtype_spec, parse_dates=["time"])
    annotations = COCO(output_dir / "cvat_instances_default.json")
    df = handle_nuc_cyto(experiment, task_df, annotations, "nucleus", "soma")
    df.to_csv(output_dir / "synthetic_nuc_cyto_measurements.csv", index=False)


def test_nuc_cyto(output_dir: Path):
    client = new_client_from_config(config)
    if get_project(client, "synthetic_nuc_cyto") is None:
        upload_synthetic_nuc_cyto_data(output_dir)
    handle_export(client, "synthetic_nuc_cyto", output_dir, "COCO 1.0")
    measure_synthetic_nuc_cyto_data(output_dir)
