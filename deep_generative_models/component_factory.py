from typing import Dict, Any, Type, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.arguments import ArgumentValidator
from deep_generative_models.arguments import ArgumentError, MissingArgument, InvalidArgument
from deep_generative_models.metadata import Metadata


class ComponentFactory(ArgumentValidator):

    def dependencies(self, arguments: Configuration) -> List[str]:
        return []

    def mandatory_architecture_arguments(self) -> List[str]:
        return []

    def validate_architecture_arguments(self, architecture_arguments: Configuration) -> None:
        for mandatory_architecture_argument in self.mandatory_architecture_arguments():
            if mandatory_architecture_argument not in architecture_arguments:
                raise MissingArchitectureArgument(mandatory_architecture_argument)

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        raise NotImplementedError


class MultiComponentFactory(ComponentFactory):
    factory_by_name: Dict[str, ComponentFactory]
    
    def __init__(self, factory_by_name: Dict[str, ComponentFactory]) -> None:
        self.factory_by_name = factory_by_name

    def create_other(self, other_name: str, architecture: Architecture, metadata: Metadata,
                     other_arguments: Configuration) -> Any:
        other_factory = self.factory_by_name[other_name]

        try:
            other_factory.validate_architecture_arguments(architecture.arguments)
        except MissingArchitectureArgument as e:
            raise Exception("Missing architecture argument '{}' while creating other component '{}'".format(e.name, other_name))

        try:
            other_factory.validate_arguments(other_arguments)
        except MissingArgument as e:
            raise Exception("Missing argument '{}' while creating other component '{}'".format(e.name, other_name))
        except InvalidArgument as e:
            raise Exception("Invalid argument '{}' while creating other component '{}'".format(e.name, other_name))

        return other_factory.create(architecture, metadata, other_arguments)

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        raise NotImplementedError


class ComponentFactoryFromClass(ComponentFactory):

    wrapped_class: Type
    optional_class_arguments: List[str]

    def __init__(self, wrapped_class: Type, optional_class_arguments: List[str] = ()):
        self.wrapped_class = wrapped_class
        self.optional_class_arguments = optional_class_arguments

    def optional_arguments(self) -> List[str]:
        return self.optional_class_arguments

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return self.wrapped_class(**arguments.get_all_defined(self.optional_class_arguments))


class MissingArchitectureArgument(ArgumentError):
    pass


