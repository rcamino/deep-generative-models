from typing import Dict, Any, Type, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class Factory:

    def dependencies(self, arguments: Configuration) -> List[str]:
        return []

    def mandatory_architecture_arguments(self) -> List[str]:
        return []

    def mandatory_arguments(self) -> List[str]:
        return []

    def optional_arguments(self) -> List[str]:
        return []

    def validate_arguments(self, architecture_arguments: Configuration, arguments: Configuration) -> None:
        # architecture mandatory arguments
        for mandatory_architecture_argument in self.mandatory_architecture_arguments():
            if mandatory_architecture_argument not in architecture_arguments:
                raise MissingArchitectureArgument(mandatory_architecture_argument)

        # keep the remaining arguments here
        remaining_arguments = set(arguments.keys())

        # mandatory arguments
        for mandatory_argument in self.mandatory_arguments():
            if mandatory_argument in remaining_arguments:
                remaining_arguments.remove(mandatory_argument)
            else:
                raise MissingFactoryArgument(mandatory_argument)

        # optional arguments
        for optional_argument in self.optional_arguments():
            if optional_argument in remaining_arguments:
                remaining_arguments.remove(optional_argument)

        # invalid arguments
        for remaining_argument in remaining_arguments:
            raise InvalidFactoryArgument(remaining_argument)

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        raise NotImplementedError


class MultiFactory(Factory):
    factory_by_name: Dict[str, Factory]
    
    def __init__(self, factory_by_name: Dict[str, Factory]) -> None:
        self.factory_by_name = factory_by_name

    def create_other(self, other_name: str, architecture: Architecture, metadata: Metadata,
                     other_arguments: Configuration) -> Any:
        other_factory = self.factory_by_name[other_name]

        try:
            other_factory.validate_arguments(architecture.arguments, other_arguments)
        except MissingArchitectureArgument as e:
            raise Exception("Missing architecture argument '{}' while creating other component '{}'".format(e.name, other_name))
        except MissingFactoryArgument as e:
            raise Exception("Missing argument '{}' while creating other component '{}'".format(e.name, other_name))
        except InvalidFactoryArgument as e:
            raise Exception("Invalid argument '{}' while creating other component '{}'".format(e.name, other_name))

        return other_factory.create(architecture, metadata, other_arguments)

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        raise NotImplementedError


class ClassFactoryWrapper(Factory):

    wrapped_class: Type
    optional_class_arguments: List[str]

    def __init__(self, wrapped_class: Type, optional_class_arguments: List[str] = ()):
        self.wrapped_class = wrapped_class
        self.optional_class_arguments = optional_class_arguments

    def optional_arguments(self) -> List[str]:
        return self.optional_class_arguments

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return self.wrapped_class(**arguments.get_all_defined(self.optional_class_arguments))


class FactoryArgumentError(Exception):
    
    name: str

    def __init__(self, name: str) -> None:
        super(FactoryArgumentError, self).__init__()
        self.name = name


class MissingArchitectureArgument(FactoryArgumentError):
    pass


class MissingFactoryArgument(FactoryArgumentError):
    pass


class InvalidFactoryArgument(FactoryArgumentError):
    pass
