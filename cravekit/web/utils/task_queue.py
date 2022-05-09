import gc
import threading
import time
import uuid

from loguru import logger

from cravekit.web.other.removebg import process_remove_bg


class TaskQueue(threading.Thread):
    def __init__(self):
        super().__init__()
        self.jobs = {}
        self.completed_jobs = {}

    def run(self):
        unused_completed_jobs_timer = time.time()
        while True:
            # Clear unused completed jobs
            if time.time() - unused_completed_jobs_timer > 60:
                self.clear_old_completed_jobs()
                unused_completed_jobs_timer = time.time()

            if len(self.jobs.keys()) >= 1:
                id = list(self.jobs.keys())[0]
                data = self.jobs[id]
                response = process_remove_bg(data[0], data[1], data[2], data[3])  # TODO fix it
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
        if len(self.completed_jobs.keys()) >= 1:
            for job_id in self.completed_jobs.keys():
                job_finished_time = self.completed_jobs[job_id][1]
                if time.time() - job_finished_time > 3600:
                    try:
                        del self.completed_jobs[job_id]
                    except KeyError or NameError as e:
                        logger.error(f"Something went wrong with Task Queue: {str(e)}")
            gc.collect()

    def job_status(self, id):
        """
        :param id: job id
        :return: Job status
        """
        if id in self.completed_jobs.keys():
            return "finished"
        elif id in self.jobs.keys():
            return "wait"
        else:
            return "not_found"

    def job_result(self, id):
        """
        :param id: Job id
        :return: Result for this task
        """
        if id in self.completed_jobs.keys():
            data = self.completed_jobs[id][0]
            try:
                del self.completed_jobs[id]
            except BaseException:
                pass
            return data
        else:
            return False

    def job_create(self, data: list):
        """
        Send job to queue
        :param data: Job data
        :return: Job id
        """
        id = uuid.uuid4().hex
        self.jobs[id] = data
        return id
