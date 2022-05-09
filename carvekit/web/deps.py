from carvekit.web.schemas.config import WebAPIConfig
from carvekit.web.utils.init_utils import init_config
from carvekit.web.utils.task_queue import MLProcessor

config: WebAPIConfig = init_config()
ml_processor = MLProcessor(api_config=config)

