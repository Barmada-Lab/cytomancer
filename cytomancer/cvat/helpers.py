import time
from itertools import groupby

import numpy as np
from cvat_sdk import Client, Config
from skimage.measure import regionprops

from cytomancer.config import CytomancerConfig


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


def _mask_to_rle(mask: np.ndarray) -> list[int]:
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.flatten())):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return counts


def get_rles(labelled_arr: np.ndarray):
    for props in regionprops(labelled_arr):
        id = props.label
        mask = labelled_arr == id
        top, left, bottom, right = props.bbox
        rle = _mask_to_rle(mask[top:bottom, left:right])
        rle += [left, top, right - 1, bottom - 1]
        yield (id, rle)


def create_project(client: Client, project_name: str):
    """
    Creates a new project with the given name
    """
    project = client.projects.create({"name": project_name})
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
    labels = {
        label.name: label.id
        for label in client.projects.retrieve(project_id).get_labels()
    }
    return labels
