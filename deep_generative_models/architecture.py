from typing import Union, List

from torch.nn import Module
from torch.optim.optimizer import Optimizer

from deep_generative_models.configuration import Configuration
from deep_generative_models.dictionary import Dictionary
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.models.initialization import initialize_module
from deep_generative_models.optimizers.wrapped_optimizer import WrappedOptimizer


# I don't know how to name this...
Component = Union[Module, Optimizer, WrappedOptimizer]


class Architecture(Dictionary[Component]):
    arguments: Configuration

    def __init__(self, arguments: Configuration):
        super(Architecture, self).__init__()
        self.__dict__["arguments"] = arguments  # to avoid the wrapped dictionary

    def to_gpu_if_available(self) -> None:
        for name, component in self.items():
            if isinstance(component, Module):  # skip optimizers
                self[name] = to_gpu_if_available(component)

    def to_cpu_if_was_in_gpu(self) -> None:
        for name, component in self.items():
            if isinstance(component, Module):  # skip optimizers
                self[name] = to_cpu_if_was_in_gpu(component)

    def initialize(self):
        for component in self.values():
            if isinstance(component, Module):  # skip optimizers
                initialize_module(component)


class ArchitectureConfigurationValidator:

    def mandatory_architecture_arguments(self) -> List[str]:
        return []

    def mandatory_architecture_components(self) -> List[str]:
        return []

    def validate_architecture_configuration(self, architecture_configuration: Configuration) -> None:
        # mandatory arguments
        defined_arguments = set(architecture_configuration.arguments.keys())
        for mandatory_argument in self.mandatory_architecture_arguments():
            if mandatory_argument not in defined_arguments:
                raise Exception("Missing architecture argument '{}'".format(mandatory_argument))

        # mandatory components
        defined_components = set(architecture_configuration.components.keys())
        for mandatory_component in self.mandatory_architecture_components():
            if mandatory_component not in defined_components:
                raise Exception("Missing architecture component '{}'".format(mandatory_component))
