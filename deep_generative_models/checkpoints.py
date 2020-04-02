import os
import time

import torch

from typing import Optional, Dict, Any

from deep_generative_models.architecture import Architecture
from deep_generative_models.commandline import DelayedKeyboardInterrupt


Checkpoint = Dict[str, Any]


class Checkpoints(object):
    last_flush_time: Optional[float]
    kept_checkpoint: Optional[Checkpoint]

    def __init__(self) -> None:
        self.last_flush_time = None
        self.kept_checkpoint = None

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def load(path: str) -> Checkpoint:
        # ignore the location
        return torch.load(path, map_location=lambda storage, loc: storage)

    @staticmethod
    def load_states(sources: Checkpoint, targets: Architecture) -> None:
        for name, target in targets.items():
            target.load_state_dict(sources[name])

    @staticmethod
    def extract_states(sources: Architecture) -> Checkpoint:
        targets = {}
        for name, source in sources.items():
            targets[name] = source.state_dict()
        return targets

    def delayed_save(self, checkpoint: Checkpoint, path: str, max_delay: int, keep: bool = False) -> None:
        now = time.time()

        # if this is the first save the time from last save is zero
        if self.last_flush_time is None:
            self.last_flush_time = now
            seconds_without_save = 0

        # if not calculate the time from last save
        else:
            seconds_without_save = now - self.last_flush_time

        # if too much time passed from last save
        if seconds_without_save > max_delay:
            # save this one
            self.save(checkpoint, path, ignore_kept=True)
            self.last_flush_time = now
            self.kept_checkpoint = None

        # if not too much time passed but should be kept
        elif keep:
            self.kept_checkpoint = checkpoint

    def save(self, checkpoint: Checkpoint, path: str, ignore_kept: bool = True) -> None:
        with DelayedKeyboardInterrupt():
            # if kept should be ignored this one is used
            if ignore_kept:
                torch.save(checkpoint, path)

            # if there is one kept and should be used
            elif self.kept_checkpoint is not None:
                torch.save(self.kept_checkpoint, path)
