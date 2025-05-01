from contextlib import contextmanager

from distributed import Client, LocalCluster

from cytomancer.config import config


@contextmanager
def dask_client():
    with (
        LocalCluster(
            n_workers=config.dask_n_workers,
            threads_per_worker=config.dask_threads_per_worker,
            local_directory=config.scratch_dir
        ) as cluster,
        Client(cluster) as client,
    ):
        yield client
