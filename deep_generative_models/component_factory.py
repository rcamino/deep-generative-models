from typing import Dict, Any, Type, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.arguments import MissingArgument, InvalidArgument
from deep_generative_models.configuration import Configuration
from deep_generative_models.arguments import ArgumentValidator
from deep_generative_models.metadata import Metadata


class ComponentFactory(ArgumentValidator):

    def dependencies(self, arguments: Configuration) -> List[str]:
        return []

    def mandatory_architecture_arguments(self) -> List[str]:
        return []

    def validate_and_create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # validate the architecture arguments
        for mandatory_architecture_argument in self.mandatory_architecture_arguments():
            if mandatory_architecture_argument not in architecture.arguments:
                raise Exception("Missing architecture argument '{}' while creating component.".format(
                    mandatory_architecture_argument))

        # validate the component arguments
        try:
            self.validate_arguments(arguments)
        except MissingArgument as e:
            raise Exception("Missing argument '{}' while creating component".format(e.name))
        except InvalidArgument as e:
            raise Exception("Invalid argument '{}' while creating component.".format(e.name))

        # create the component
        return self.create(architecture, metadata, arguments)

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        raise NotImplementedError


class MultiComponentFactory(ComponentFactory):
    factory_by_name: Dict[str, ComponentFactory]

    def __init__(self, factory_by_name: Dict[str, ComponentFactory]) -> None:
        self.factory_by_name = factory_by_name

    def create_other(self, other_name: str, architecture: Architecture, metadata: Metadata,
                     other_arguments: Configuration) -> Any:
        return self.factory_by_name[other_name].validate_and_create(architecture, metadata, other_arguments)

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
