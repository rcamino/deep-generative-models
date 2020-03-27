from typing import Dict, Any, Type, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class Factory:

    @staticmethod
    def dependencies(configuration: Configuration) -> List[str]:
        return []

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        raise NotImplementedError


class MultiFactory(Factory):
    factory_by_name: Dict[str, Factory]
    
    def __init__(self, factory_by_name: Dict[str, Factory]) -> None:
        self.factory_by_name = factory_by_name

    def create_other(self, other_name: str, architecture: Architecture, metadata: Metadata,
                     global_configuration: Configuration, other_configuration: Configuration) -> Any:
        other_factory = self.factory_by_name[other_name]
        return other_factory.create(architecture, metadata, global_configuration, other_configuration)

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        raise NotImplementedError


class ClassFactoryWrapper(Factory):

    wrapped_class: Type

    def __init__(self, wrapped_class: Type):
        self.wrapped_class = wrapped_class

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return self.wrapped_class(**configuration)
