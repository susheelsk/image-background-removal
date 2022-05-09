import gc
import threading
import time
import uuid
from typing import Optional

from loguru import logger

from carvekit.api.interface import Interface
from carvekit.web.schemas.config import WebAPIConfig
from carvekit.web.utils.init_utils import init_interface
from carvekit.web.other.removebg import process_remove_bg


class MLProcessor(threading.Thread):
    """Simple ml task queue processor"""
    def __init__(self, api_config: WebAPIConfig):
        super().__init__()
        self.api_config = api_config
        self.interface: Optional[Interface] = None
        self.jobs = {}
        self.completed_jobs = {}

    def run(self):
        """Starts listening for new jobs."""
        unused_completed_jobs_timer = time.time()
        if self.interface is None:
            self.interface = init_interface(self.api_config)
        while True:
            # Clear unused completed jobs every hour
            if time.time() - unused_completed_jobs_timer > 60:
                self.clear_old_completed_jobs()
                unused_completed_jobs_timer = time.time()

            if len(self.jobs.keys()) >= 1:
                id = list(self.jobs.keys())[0]
                data = self.jobs[id]
                # TODO add pydantic scheme here
                response = process_remove_bg(self.interface, data[0], data[1], data[2], data[3])
                self.completed_jobs[id] = [response, time.time()]
                try:
                    del self.jobs[id]
                except KeyError or NameError as e:
                    logger.error(f"Something went wrong with Task Queue: {str(e)}")
                gc.collect()
            else:
                time.sleep(1)
                continue

    def clear_old_completed_jobs(self):
        """Clears old completed jobs"""

        if len(self.completed_jobs.keys()) >= 1:
            for job_id in self.completed_jobs.keys():
                job_finished_time = self.completed_jobs[job_id][1]
                if time.time() - job_finished_time > 3600:
                    try:
                        del self.completed_jobs[job_id]
                    except KeyError or NameError as e:
                        logger.error(f"Something went wrong with Task Queue: {str(e)}")
            gc.collect()

    def job_status(self, id: str) -> str:
        """
        Returns current job status

        Args:
            id: id of the job

        Returns:
            Current job status for specified id. Job status can be [finished, wait, not_found]
        """
        if id in self.completed_jobs.keys():
            return "finished"
        elif id in self.jobs.keys():
            return "wait"
        else:
            return "not_found"

    def job_result(self, id: str):
        """
        Returns job processing result.

        Args:
            id: id of the job

        Returns:
            job processing result.
        """
        if id in self.completed_jobs.keys():
            data = self.completed_jobs[id][0]
            try:
                del self.completed_jobs[id]
            except KeyError or NameError:
                pass
            return data
        else:
            return False

    def job_create(self, data: list):
        """
        Send job to ML Processor

        Args:
            data: data object
        """
        if self.is_alive() is False:
            self.start()
        id = uuid.uuid4().hex
        self.jobs[id] = data
        return id
