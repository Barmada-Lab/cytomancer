from skimage.measure import regionprops
from itertools import groupby
from dataclasses import dataclass
import time

from cvat_sdk import Client, Config
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np

from cytomancer.config import CytomancerConfig
from cytomancer.experiment import Axes


def exponential_backoff(max_retries=5, base_delay=0.1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retries >= max_retries:
                        raise e
                    retries += 1
                    delay *= 2
                    time.sleep(delay)
        return wrapper
    return decorator


def new_client_from_config(config: CytomancerConfig):
    client = Client(url=config.cvat_url, config=Config(verify_ssl=False))
    client.login((config.cvat_username, config.cvat_password))
    client.organization_slug = config.cvat_org
    return client


def test_cvat_credentials(cvat_url, cvat_username, cvat_password):
    """
    Test the connection to a CVAT server.

    Args:
        cvat_url (str): The URL of the CVAT server.
        cvat_username (str): The username to use for authentication.
        cvat_password (str): The password to use for authentication.

    Returns:
        bool: True if the connection was successful, False otherwise.
    """
    from cvat_sdk import Client
    from cvat_sdk.exceptions import ApiException
    client = Client(url=cvat_url)
    try:
        client.login((cvat_username, cvat_password))
        return True
    except ApiException as e:
        print(f"Error: {e.body}")
        return False


# ex. field-1|region-B02|channel-GFP:RFP:Cy5|time-1:2:3:4:5:6:7:8:9:10
def _fmt_coord_selector_str(label, coord_arr):
    arr = np.atleast_1d(coord_arr)
    if label == "time":
        arr = arr.astype("long")
    if np.issubdtype(arr.dtype, np.str_):
        for value in arr:
            assert FIELD_DELIM not in value, f"{label} value {value} is invalid; contains a '|'; rename and try again"
            assert VALUE_DELIM not in value, f"{label} value {value} is invalid; contains a ':'; rename and try again"

    return f"{label}{FIELD_VALUE_DELIM}" + VALUE_DELIM.join(map(str, arr))


def coord_selector(arr: xr.DataArray) -> str:
    """Derives a string-formatted selector from an array's coordinates."""
    coords = sorted(arr.coords.items())
    filtered = filter(lambda coord: coord[0] not in ["x", "y"], coords)
    return FIELD_DELIM.join([
        _fmt_coord_selector_str(axis.value, coord.values) for axis, coord in filtered  # type: ignore
    ])



def mask_to_rle(mask: np.ndarray) -> list[int]:
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.flatten())):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return counts


def get_rles(labelled_arr: np.ndarray):
    rles = []
    for props in regionprops(labelled_arr):
        id = props.label
        mask = labelled_arr == id
        top, left, bottom, right = props.bbox
        rle = mask_to_rle(mask[top:bottom, left:right])
        rle += [left, top, right-1, bottom-1]

        left, top, right, bottom = rle[-4:]
        patch_height, patch_width = (bottom - top + 1, right - left + 1)
        patch_mask = rle_to_mask(rle[:-4], patch_width, patch_height)

        assert np.all(patch_mask == mask[top:bottom+1, left:right+1])
        rles.append((id, rle))
    return rles


def enumerate_rois(client: Client, project_id: int, progress: bool = False):
    """
    enumerates all ROIs in a project on a frame-by-frame basis
    """
    tasks = client.projects.retrieve(project_id).get_tasks()
    if progress:
        tasks = tqdm(tasks)
    for task_meta in tasks:
        jobs = task_meta.get_jobs()
        job_id = jobs[0].id  # we assume there is only one job per task
        job_metadata = client.jobs.retrieve(job_id).get_meta()
        frames = job_metadata.frames
        height, width = frames[0].height, frames[0].width
        anno_table = task_meta.get_annotations()
        obj_arr, label_arr = get_obj_arr_and_labels(anno_table, len(frames), height, width)
        for frame, obj_frame, label_frame in zip(frames, obj_arr, label_arr):
            yield frame.name, obj_frame, label_frame


def create_project(client: Client, project_name: str):
    """
    Creates a new project with the given name
    """
    project = client.projects.create(dict(name=project_name))
    return project


def get_project(client: Client, project_name: str):
    """
    Returns a project with the given name, or None if no such project exists.
    """
    for project in client.projects.list():
        if project.name == project_name:
            return project
    return None


def get_project_label_map(client: Client, project_id: int):
    """
    Returns a list of all labelled arrays for a given project.
    """
    labels = {label.name: label.id for label in client.projects.retrieve(project_id).get_labels()}
    return labels
