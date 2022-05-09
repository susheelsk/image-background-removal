import time

from cravekit.web.utils.task_queue import TaskQueue
from cravekit.web.schemas.config import WebAPIConfig
from cravekit.web.utils.init_config import init_config

config: WebAPIConfig = init_config()
queue = TaskQueue()
start_time: float = time.time()
