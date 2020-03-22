import os
import time

import torch

from torch.nn import Module

from typing import Optional, Dict, Any

from deep_generative_models.architecture import Architecture
from deep_generative_models.commandline import DelayedKeyboardInterrupt


Checkpoint = Dict[str, Any]


class Checkpoints(object):
    path: str
    max_delay: int
    last_flush_time: Optional[float]
    kept_checkpoint: Optional[Checkpoint]

    def __init__(self, path: str, max_delay: int) -> None:
        self.path = path
        self.max_seconds_without_save = max_delay

        self.last_flush_time = None
        self.kept_checkpoint = None

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def load(self) -> Checkpoint:
        # ignore the location
        return torch.load(self.path, map_location=lambda storage, loc: storage)

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

    def delayed_save(self, checkpoint: Checkpoint, keep: bool = False) -> None:
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
            # save this one
            self.save(checkpoint, ignore_kept=True)
            self.last_flush_time = now
            self.kept_checkpoint = None

        # if not too much time passed but should be kept
        elif keep:
            self.kept_checkpoint = checkpoint

    def save(self, checkpoint: Checkpoint, ignore_kept: bool = True) -> None:
        with DelayedKeyboardInterrupt():
            # if kept should be ignored this one is used
            if ignore_kept:
                torch.save(checkpoint, self.path)

            # if there is one kept and should be used
            elif self.kept_checkpoint is not None:
                torch.save(self.kept_checkpoint, self.path)
