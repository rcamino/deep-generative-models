import time

import torch

from typing import Optional

from deep_generative_models.type_aliases import Checkpoint, ModuleDictionary
from deep_generative_models.logger import Logger
from deep_generative_models.commandline import DelayedKeyboardInterrupt


class Saver(object):
    path: str
    modules: ModuleDictionary
    logger: Logger
    max_delay: int
    last_flush_time: Optional[float]
    kept_checkpoint: Optional[Checkpoint]
    
    def __init__(self, path: str, modules: ModuleDictionary, logger: Logger, max_delay: int) -> None:
        self.path = path
        self.modules = modules
        self.logger = logger
        self.max_seconds_without_save = max_delay
        
        self.last_flush_time = None
        self.kept_checkpoint = None
    
    def delayed_save(self, keep_parameters: bool = False, additional: Optional[Checkpoint] = None) -> None:
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
            self.kept_checkpoint = None

        # if not too much time passed but parameters should be kept
        elif keep_parameters:
            self.kept_checkpoint = {}
            for module_name, module in self.modules.items():
                self.kept_checkpoint[module_name] = module.state_dict()
            if additional is not None:
                self.kept_checkpoint.update(additional)

    def save(self, ignore_kept: bool = True, additional: Optional[Checkpoint] = None) -> None:
        with DelayedKeyboardInterrupt():
            # if kept parameters should be ignored the current model parameters are used
            if ignore_kept:
                checkpoint = {}
                for module_name, module in self.modules.items():
                    checkpoint[module_name] = module.state_dict()
                if additional is not None:
                    checkpoint.update(additional)
                torch.save(checkpoint, self.path)

            # if kept parameters should be used and they are defined
            elif self.kept_checkpoint is not None:
                torch.save(self.kept_checkpoint, self.path)

            # flush all the logs
            self.logger.flush()
