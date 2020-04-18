import argparse
import csv
import time

from multiprocessing import Process, Queue, log_to_stderr

from queue import Empty
from typing import Type, List

from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.task import Task


EVENT_TYPE_EXIT = "exit"
EVENT_TYPE_WRITE = "write"


logger = log_to_stderr()


class MultiProcessTaskWorker(Task):
    write_queue: Queue
    worker_number: int
    worker_configuration: Configuration

    @classmethod
    def output_fields(cls) -> List[str]:
        raise NotImplementedError

    def __init__(self, write_queue: Queue, worker_number: int, worker_configuration: Configuration) -> None:
        super(MultiProcessTaskWorker, self).__init__(logger)

        self.write_queue = write_queue
        self.worker_number = worker_number
        self.worker_configuration = worker_configuration

    def send_output(self, output: dict) -> None:
        self.write_queue.put({"event_type": EVENT_TYPE_WRITE, "row": output})

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError


class SimpleMultiProcessTaskWorker(MultiProcessTaskWorker):

    @classmethod
    def output_fields(cls) -> List[str]:
        return ["has_error", "error", "worker"]

    def run(self, configuration: Configuration) -> None:
        from deep_generative_models.tasks.runner import TaskRunner  # import here to avoid circular dependency
        task_runner = TaskRunner()

        output = {"has_error": False, "worker": self.worker_number}

        try:
            task_runner.run(configuration)
        except Exception as e:
            output["has_error"] = True
            output["error"] = repr(e)

        self.send_output(output)


class MultiProcessTaskRunner(Task):
    task_worker_class: Type[MultiProcessTaskWorker]
    
    def __init__(self, task_worker_class: Type[MultiProcessTaskWorker]):
        super(MultiProcessTaskRunner, self).__init__()
        self.task_worker_class = task_worker_class

    def mandatory_arguments(self) -> List[str]:
        return ["workers", "inputs", "output"]

    def optional_arguments(self) -> List[str]:
        return ["log_every"]

    def run(self, configuration: Configuration) -> None:
        read_queue = Queue()
        write_queue = Queue()

        # queue all the task configurations
        for task_configuration_path in configuration.inputs:
            read_queue.put(task_configuration_path)

        # write worker: we will write in the output file using only one process and a queue
        output_path = create_parent_directories_if_needed(configuration.output)
        output_fields = self.task_worker_class.output_fields()
        write_process = Process(target=write_worker, args=(write_queue, output_path, output_fields))
        write_process.start()

        # additional process to log the remaining tasks
        count_process = Process(
            target=count_worker,
            args=(read_queue, configuration.get("log_every", 5)))
        count_process.start()
        # we don't need to join this one

        # workers: we will process tasks in parallel
        if type(configuration.workers) == int:
            worker_configurations = [{} for _ in range(configuration.workers)]
        elif type(configuration.workers) == list:
            worker_configurations = configuration.workers
        else:
            raise Exception("Invalid worker configuration.")

        worker_processes = []
        for worker_number, worker_configuration in enumerate(worker_configurations):
            worker_process = Process(
                target=task_worker_wrapper,
                args=(self.task_worker_class, worker_number, worker_configuration, read_queue, write_queue))
            worker_process.start()
            worker_processes.append(worker_process)

        # wait for all the workers to finish
        logger.info("Waiting for the workers...")
        for worker_process in worker_processes:
            worker_process.join()
        logger.info("Workers finished.")

        # the workers stopped queuing rows
        # add to stop event for the writing worker
        write_queue.put({"event_type": EVENT_TYPE_EXIT})

        # wait until the writing worker actually stops
        write_process.join()


def task_worker_wrapper(task_worker_class: Type[MultiProcessTaskWorker],
                        worker_number: int,
                        worker_configuration: Configuration,
                        read_queue: Queue,
                        write_queue: Queue
                        ) -> None:
    logger.info("Worker {:d} started...".format(worker_number))

    # create the task worker
    task_worker = task_worker_class(write_queue, worker_number, worker_configuration)

    # while there are more tasks in the queue
    while True:
        # get the next task if possible
        try:
            task_configuration_path = read_queue.get(block=True, timeout=1)
            logger.debug("Next task on worker {:d}".format(worker_number))

        # no more tasks in the queue
        except Empty:
            logger.info("No more tasks for worker {:d}.".format(worker_number))
            break

        # run the next task
        task_worker.run(load_configuration(task_configuration_path))

    logger.info("Worker {:d} finished.".format(worker_number))


def count_worker(queue: Queue, log_every: int = 5):
    size = queue.qsize()
    last_size = size
    logger.info("{:d} remaining...".format(size))
    while size > 0:
        time.sleep(log_every)
        size = queue.qsize()
        processed = last_size - size
        logger.info("{:d} processed, {:d} remaining...".format(processed, size))
        last_size = size


def write_worker(queue: Queue, file_path: str, field_names: List[str]):
    logger.info("Writing started...")

    f = open(file_path, "w")

    writer = csv.DictWriter(f, field_names)
    writer.writeheader()

    while True:
        # wait until there is a new event
        event = queue.get(block=True)

        # write event
        if event["event_type"] == EVENT_TYPE_WRITE:
            writer.writerow(event["row"])
            f.flush()
        # exit event
        elif event["event_type"] == EVENT_TYPE_EXIT:
            break
        # something went wrong
        else:
            raise Exception("Invalid event type '{}'".format(event["event_type"]))

    f.close()

    logger.info("Writing finished.")


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Run tasks in multi-processing mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    MultiProcessTaskRunner(SimpleMultiProcessTaskWorker).timed_run(load_configuration(options.configuration))
