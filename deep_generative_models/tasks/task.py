import logging
import time

from logging import Logger
from typing import Optional

from deep_generative_models.configuration import Configuration
from deep_generative_models.arguments import ArgumentValidator


class Task(ArgumentValidator):
    _logger: Optional[Logger]

    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = logging.getLogger(__name__)
        return self._logger

    def timed_run(self, configuration: Configuration) -> None:
        if self._logger is None:
            logging.basicConfig(**configuration.get("logging", default={}, transform_default=False))

        self.logger.info("Starting task...")
        start_time = time.time()
        self.run(configuration)
        self.logger.info("Task ended in {:02f}s.".format(time.time() - start_time))

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError
