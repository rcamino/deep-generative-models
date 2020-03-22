from typing import Dict, Any, Type

from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class Factory:

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        raise NotImplementedError


class MultiFactory(Factory):
    factory_by_name: Dict[str, Factory]
    
    def __init__(self, factory_by_name: Dict[str, Factory]) -> None:
        self.factory_by_name = factory_by_name

    def create_other(self, other_name: str, metadata: Metadata, global_configuration: Configuration,
                     other_configuration: Configuration) -> Any:
        other_factory = self.factory_by_name[other_name]
        return other_factory.create(metadata, global_configuration, other_configuration)

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        raise NotImplementedError


class ClassFactoryWrapper(Factory):

    wrapped_class: Type

    def __init__(self, wrapped_class: Type):
        self.wrapped_class = wrapped_class

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        arguments = configuration.get("arguments", default=[], transform_default=False)
        keyword_arguments = configuration.get("keyword_arguments", default={}, transform_default=False)
        return self.wrapped_class(*arguments, **keyword_arguments)
