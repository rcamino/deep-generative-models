import torch

from torch import Tensor
from torch.nn import Module

from typing import Union, List


CanBeMovedToGPU = Union[Tensor, Module]


def to_gpu_if_available(*targets: CanBeMovedToGPU) -> Union[CanBeMovedToGPU, List[CanBeMovedToGPU]]:
    if len(targets) == 0:
        return []
    if torch.cuda.is_available():
        targets = [target.cuda() if target is not None else None for target in targets]
    if len(targets) == 1:
        return targets[0]
    return targets


def to_cpu_if_was_in_gpu(*targets: CanBeMovedToGPU) -> Union[CanBeMovedToGPU, List[CanBeMovedToGPU]]:
    if len(targets) == 0:
        return []
    if torch.cuda.is_available():
        targets = [target.cpu() if target is not None else None for target in targets]
    if len(targets) == 1:
        return targets[0]
    return targets
