from typing import Union

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from deep_generative_models.configuration import Configuration
from deep_generative_models.dictionary import Dictionary
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.models.initialization import initialize_module


# I don't know how to name this...
Component = Union[Module, Optimizer]


class Architecture(Dictionary[Component]):

    arguments: Configuration

    def __init__(self, arguments: Configuration):
        super(Architecture, self).__init__()
        self.__dict__["arguments"] = arguments  # to avoid the wrapped dictionary

    def to_gpu_if_available(self) -> None:
        for name, module in self.items():
            self[name] = to_gpu_if_available(module)

    def to_cpu_if_was_in_gpu(self) -> None:
        for name, module in self.items():
            self[name] = to_cpu_if_was_in_gpu(module)

    def initialize(self):
        for module in self.values():
            initialize_module(module)
