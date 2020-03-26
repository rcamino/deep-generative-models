import os
import time

import torch

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
