import time

import torch

from torch.nn import Module

from typing import Optional, Dict, Any

from deep_generative_models.architecture import Architecture
from deep_generative_models.logger import Logger
from deep_generative_models.commandline import DelayedKeyboardInterrupt


Checkpoint = Dict[str, Any]


class Checkpoints(object):
    path: str
    architecture: Architecture
    logger: Logger
    max_delay: int
    last_flush_time: Optional[float]
    kept_checkpoint: Optional[Checkpoint]

    def __init__(self, path: str, architecture: Architecture, logger: Logger, max_delay: int) -> None:
        self.path = path
        self.architecture = architecture
        self.logger = logger
        self.max_seconds_without_save = max_delay

        self.last_flush_time = None
        self.kept_checkpoint = None

    def load_architecture(self, architecture: Architecture, checkpoint: Checkpoint) -> None:
        for module_name, module in architecture.items():
            self.load_module(module, checkpoint, module_name)

    @staticmethod
    def load_module(module: Module, checkpoint: Checkpoint, model_name: str) -> None:
        assert model_name in checkpoint, "'{}' not found in checkpoint.".format(model_name)
        module.load_state_dict(checkpoint[model_name])

    def extract_from_architecture(self, modules: Architecture) -> Checkpoint:
        checkpoint = {}
        for module_name, module in modules.items():
            self.extract_from_module(module_name, module, checkpoint)
        return checkpoint

    @staticmethod
    def extract_from_module(module_name: str, module: Module, kept_checkpoint: Checkpoint) -> None:
        kept_checkpoint[module_name] = module.state_dict()

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
            self.kept_checkpoint = self.extract_from_architecture(self.architecture)
            if additional is not None:
                self.kept_checkpoint.update(additional)

    def save(self, ignore_kept: bool = True, additional: Optional[Checkpoint] = None) -> None:
        with DelayedKeyboardInterrupt():
            # if kept parameters should be ignored the current model parameters are used
            if ignore_kept:
                checkpoint = self.extract_from_architecture(self.architecture)
                if additional is not None:
                    checkpoint.update(additional)
                torch.save(checkpoint, self.path)

            # if kept parameters should be used and they are defined
            elif self.kept_checkpoint is not None:
                torch.save(self.kept_checkpoint, self.path)

            # flush all the logs
            self.logger.flush()
