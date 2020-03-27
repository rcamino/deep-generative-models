from typing import Type, Any, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.metadata import Metadata


class OptimizerFactory(Factory):

    optimizer_class: Type

    def __init__(self, optimizer_class: Type):
        self.optimizer_class = optimizer_class

    @staticmethod
    def dependencies(configuration: Configuration) -> List[str]:
        return configuration.parameters

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        # collect the parameters from the indicated models
        parameters = []
        for module_name in configuration.parameters:
            parameters.extend(architecture[module_name].parameters())

        # copy the rest of the arguments
        arguments = {}
        for key, value in configuration.items():
            if key != "parameters":
                arguments[key] = value

        # create the optimizer
        return self.optimizer_class(parameters, **arguments)
