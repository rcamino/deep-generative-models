import argparse
import csv
import time
import torch

from logging import Logger
from multiprocessing import Process, Queue, log_to_stderr
from queue import Empty
from typing import Type, List, Any

from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.task import Task


EVENT_TYPE_EXIT = "exit"
EVENT_TYPE_WRITE = "write"


multiprocessing_logger = log_to_stderr()


class MultiProcessTaskWorker:
    outputs_queue: Queue
    worker_number: int
    worker_configuration: Configuration
    logger: Logger

    @classmethod
    def output_fields(cls) -> List[str]:
        raise NotImplementedError

    def __init__(self, outputs_queue: Queue, worker_number: int, worker_configuration: Configuration) -> None:
        self.outputs_queue = outputs_queue
        self.worker_number = worker_number
        self.worker_configuration = worker_configuration

        self.logger = multiprocessing_logger

    def send_output(self, output: dict) -> None:
        self.outputs_queue.put({"event_type": EVENT_TYPE_WRITE, "row": output})

    def process(self, inputs: Any) -> None:
        raise NotImplementedError


class TaskRunnerWorker(MultiProcessTaskWorker):

    @classmethod
    def output_fields(cls) -> List[str]:
        return ["path", "has_error", "error", "gpu_device", "worker", "time"]

    def process(self, inputs: Any) -> None:
        from deep_generative_models.tasks.runner import TaskRunner  # import here to avoid circular dependency
        task_runner = TaskRunner(logger=self.logger)

        start_time = time.time()
        output = {"has_error": False, "worker": self.worker_number}

        try:
            assert type(inputs) == str, "Inputs must be configuration paths."
            output["path"] = inputs
            configuration = load_configuration(inputs)

            if "gpu_device" in self.worker_configuration:
                output["gpu_device"] = self.worker_configuration.gpu_device
                with torch.cuda.device(self.worker_configuration.gpu_device):
                    task_runner.run(configuration)
            else:
                task_runner.run(configuration)
        except Exception as e:
            output["has_error"] = True
            output["error"] = repr(e)

        output["time"] = time.time() - start_time
        self.send_output(output)


class MultiProcessTaskRunner(Task):
    task_worker_class: Type[MultiProcessTaskWorker]
    
    def __init__(self, task_worker_class: Type[MultiProcessTaskWorker]):
        super(MultiProcessTaskRunner, self).__init__()
        self.task_worker_class = task_worker_class

    def mandatory_arguments(self) -> List[str]:
        return ["workers", "inputs", "output"]

    def optional_arguments(self) -> List[str]:
        return super(MultiProcessTaskRunner, self).optional_arguments() + ["log_every"]

    def run(self, configuration: Configuration) -> None:
        inputs_queue = Queue()
        outputs_queue = Queue()

        # queue all the inputs unwrapped
        for inputs in configuration.get("inputs", unwrap=True):
            inputs_queue.put(inputs)

        # outputs worker: we will write in the output file using only one process and a queue
        output_path = create_parent_directories_if_needed(configuration.output)
        output_fields = self.task_worker_class.output_fields()
        output_process = Process(target=write_worker, args=(outputs_queue, output_path, output_fields))
        output_process.start()

        # additional process to log the remaining tasks
        count_process = Process(
            target=count_worker,
            args=(inputs_queue, configuration.get("log_every", 5)))
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
                args=(self.task_worker_class, worker_number, worker_configuration, inputs_queue, outputs_queue))
            worker_process.start()
            worker_processes.append(worker_process)

        # wait for all the workers to finish
        multiprocessing_logger.info("Waiting for the workers...")
        for worker_process in worker_processes:
            worker_process.join()
        multiprocessing_logger.info("Workers finished.")

        # the workers stopped queuing rows
        # add to stop event for the writing worker
        outputs_queue.put({"event_type": EVENT_TYPE_EXIT})

        # wait until the writing worker actually stops
        output_process.join()


def task_worker_wrapper(task_worker_class: Type[MultiProcessTaskWorker],
                        worker_number: int,
                        worker_configuration: Configuration,
                        inputs_queue: Queue,
                        outputs_queue: Queue
                        ) -> None:
    multiprocessing_logger.info("Worker {:d} started...".format(worker_number))

    # create the task worker
    task_worker = task_worker_class(outputs_queue, worker_number, worker_configuration)

    # while there are more inputs in the queue
    while True:
        # get the next input if possible
        try:
            inputs = inputs_queue.get(block=True, timeout=1)
            multiprocessing_logger.debug("Next inputs for worker {:d}".format(worker_number))

        # no more inputs in the queue
        except Empty:
            multiprocessing_logger.info("No more inputs for worker {:d}.".format(worker_number))
            break

        # process the next input
        task_worker.process(inputs)

    multiprocessing_logger.info("Worker {:d} finished.".format(worker_number))


def count_worker(inputs_queue: Queue, log_every: int = 5):
    size = inputs_queue.qsize()
    last_size = size
    multiprocessing_logger.info("{:d} inputs in queue...".format(size))
    while size > 0:
        # wait for next check
        time.sleep(log_every)

        # compute how many inputs were processed from last time
        size = inputs_queue.qsize()
        processed = last_size - size

        # log only if there are changes
        if processed > 0:
            multiprocessing_logger.info("{:d} inputs removed from the queue, {:d} inputs still in the queue..."
                                        .format(processed, size))

        last_size = size


def write_worker(outputs_queue: Queue, file_path: str, field_names: List[str]):
    multiprocessing_logger.info("Output started...")

    f = open(file_path, "w")

    writer = csv.DictWriter(f, field_names)
    writer.writeheader()

    while True:
        # wait until there is a new event
        event = outputs_queue.get(block=True)

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

    multiprocessing_logger.info("Output finished.")


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Run tasks in multi-processing mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    MultiProcessTaskRunner(TaskRunnerWorker).timed_run(load_configuration(options.configuration))
