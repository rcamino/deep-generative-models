import torch

from torch import Tensor

from typing import Union, List, Any


def to_gpu_if_available(*tensors: Tensor) -> Union[Tensor, List[Tensor]]:
    if len(tensors) == 0:
        return []
    if torch.cuda.is_available():
        tensors = [tensor.cuda() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def to_cpu_if_was_in_gpu(*tensors: Tensor) -> Union[Tensor, List[Tensor]]:
    if len(tensors) == 0:
        return []
    if torch.cuda.is_available():
        tensors = [tensor.cpu() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def load_without_gpu(path: str) -> Any:
    torch.load(path, map_location=lambda storage, loc: storage)
