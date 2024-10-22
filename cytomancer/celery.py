from celery import Celery, Task
from celery.utils.log import get_task_logger

from cytomancer.config import config

logger = get_task_logger(__name__)


class CytomancerTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        pass

    def on_success(self, retval, task_id, args, kwargs):
        pass


app = Celery(
    "cytomancer",
    broker=config.celery_broker_url,
    broker_connection_retry_on_startup=True,
    task_cls=CytomancerTask,
    include=[
        "cytomancer.quant.tasks",
        "cytomancer.oneoffs.tasks",
    ],
)
