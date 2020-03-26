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
        return configuration.modules

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        parameters = []
        for module_name in configuration.modules:
            parameters.extend(architecture[module_name].parameters())
        arguments = configuration.get("arguments", default=[], transform_default=False)
        keyword_arguments = configuration.get("keyword_arguments", default={}, transform_default=False)
        return self.optimizer_class(parameters, *arguments, **keyword_arguments)
