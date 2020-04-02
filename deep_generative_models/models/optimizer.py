from typing import Type, Any, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.metadata import Metadata


class OptimizerFactory(Factory):

    optimizer_class: Type
    optional_class_arguments: List[str]

    def __init__(self, optimizer_class: Type, optional_class_arguments: List[str] = ()):
        self.optimizer_class = optimizer_class
        self.optional_class_arguments = optional_class_arguments

    def mandatory_arguments(self) -> List[str]:
        return ["parameters"]

    def optional_arguments(self) -> List[str]:
        return self.optional_class_arguments

    def dependencies(self, arguments: Configuration) -> List[str]:
        return arguments.parameters

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # collect the parameters from the indicated models
        parameters = []
        for module_name in arguments.parameters:
            parameters.extend(architecture[module_name].parameters())

        # copy the rest of the arguments
        arguments = {}
        for key, value in arguments.items():
            if key != "parameters":
                arguments[key] = value

        # create the optimizer
        return self.optimizer_class(parameters, **arguments)
