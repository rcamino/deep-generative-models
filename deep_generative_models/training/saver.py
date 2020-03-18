import time

import torch

from torch.nn import Module

from typing import Dict, Optional, Any

from deep_generative_models.training.logger import Logger
from deep_generative_models.commandline import DelayedKeyboardInterrupt


class Saver(object):
    path: str
    models_by_name: Dict[str, Module]
    logger: Logger
    max_delay: int
    last_flush_time: Optional[float]
    kept_parameters: Optional[Dict[str, Any]]
    
    def __init__(self, path: str, models_by_name: Dict[str, Module], logger: Logger, max_delay: int) -> None:
        self.path = path
        self.models_by_name = models_by_name
        self.logger = logger
        self.max_seconds_without_save = max_delay
        
        self.last_flush_time = None
        self.kept_parameters = None
    
    def delayed_save(self, keep_parameters: bool = False, additional: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()

        # if this is the first save the time from last save is zero
        if self.last_flush_time is None:
            self.last_flush_time = now
            seconds_without_save = 0

        # if not calculate the time from last save
        else:
            seconds_without_save = now - self.last_flush_time

        # if too much time passed from last save
        if seconds_without_save > self.max_seconds_without_save:
            # save the current parameters
            self.save(ignore_kept=True, additional=additional)
            self.last_flush_time = now
            self.kept_parameters = None

        # if not too much time passed but parameters should be kept
        elif keep_parameters:
            self.kept_parameters = {}
            for model_name, model in self.models_by_name.items():
                self.kept_parameters[model_name] = model.state_dict()
            if additional is not None:
                self.kept_parameters.update(additional)

    def save(self, ignore_kept: bool = True, additional: Optional[Dict[str, Any]] = None) -> None:
        with DelayedKeyboardInterrupt():
            # if kept parameters should be ignored the current model parameters are used
            if ignore_kept:
                parameters = {}
                for model_name, model in self.models_by_name.items():
                    parameters[model_name] = model.state_dict()
                if additional is not None:
                    parameters.update(additional)
                torch.save(parameters, self.path)

            # if kept parameters should be used and they are defined
            elif self.kept_parameters is not None:
                torch.save(self.kept_parameters, self.path)

            # flush all the logs
            self.logger.flush()
