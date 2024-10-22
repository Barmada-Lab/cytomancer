from contextlib import contextmanager

from distributed import Client, LocalCluster

from cytomancer.config import config


@contextmanager
def dask_client(
    n_workers=config.dask_n_workers, threads_per_worker=config.dask_threads_per_worker
):
    with LocalCluster(
        n_workers=n_workers, threads_per_worker=threads_per_worker
    ) as cluster, Client(cluster) as client:
        yield client
