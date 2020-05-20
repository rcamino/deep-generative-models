import logging
import time
import torch

from logging import Logger
from typing import Optional, List

from deep_generative_models.configuration import Configuration
from deep_generative_models.arguments import ArgumentValidator


class Task(ArgumentValidator):
    _logger: Optional[Logger]

    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger

    def optional_arguments(self) -> List[str]:
        return ["logging"]

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = logging.getLogger(__name__)
        return self._logger

    def timed_run(self, configuration: Configuration) -> None:
        logging.basicConfig(**configuration.get("logging", default={}, transform_default=False))

        self.logger.info("Starting task...")

        if torch.cuda.is_available():
            self.logger.info("Using GPU.")
        else:
            self.logger.info("Using CPU.")

        start_time = time.time()

        self.run(configuration)

        elapsed_time = time.time() - start_time
        elapsed_time_unit = "seconds"

        if elapsed_time_unit == "seconds" and elapsed_time > 60:
            elapsed_time /= 60
            elapsed_time_unit = "minutes"

        if elapsed_time_unit == "minutes" and elapsed_time > 60:
            elapsed_time /= 60
            elapsed_time_unit = "hours"

        if elapsed_time_unit == "hours" and elapsed_time > 24:
            elapsed_time /= 24
            elapsed_time_unit = "days"

        self.logger.info("Task ended in {:02f} {}.".format(elapsed_time, elapsed_time_unit))

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError
