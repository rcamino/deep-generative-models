from typing import Dict, Any, Type, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class Factory:

    def dependencies(self, configuration: Configuration) -> List[str]:
        return []

    def mandatory_global_arguments(self) -> List[str]:
        return []

    def mandatory_arguments(self) -> List[str]:
        return []

    def optional_arguments(self) -> List[str]:
        return []

    def validate_configuration(self, global_configuration: Configuration, configuration: Configuration) -> None:
        # global mandatory arguments
        for mandatory_global_argument in self.mandatory_global_arguments():
            if mandatory_global_argument not in global_configuration:
                raise MissingGlobalFactoryArgument(mandatory_global_argument)

        # keep the remaining arguments here
        remaining_arguments = set(configuration.keys())

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

        try:
            other_factory.validate_configuration(global_configuration, other_configuration)
        except MissingGlobalFactoryArgument as e:
            raise Exception("Missing global argument '{}' while creating other module '{}'".format(e.name, other_name))
        except MissingFactoryArgument as e:
            raise Exception("Missing argument '{}' while creating other module '{}'".format(e.name, other_name))
        except InvalidFactoryArgument as e:
            raise Exception("Invalid argument '{}' while creating other module '{}'".format(e.name, other_name))

        return other_factory.create(architecture, metadata, global_configuration, other_configuration)

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        raise NotImplementedError


class ClassFactoryWrapper(Factory):

    wrapped_class: Type
    optional_class_arguments: List[str]

    def __init__(self, wrapped_class: Type, optional_class_arguments: List[str] = ()):
        self.wrapped_class = wrapped_class
        self.optional_class_arguments = optional_class_arguments

    def optional_arguments(self) -> List[str]:
        return self.optional_class_arguments

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return self.wrapped_class(**configuration.get_all_defined(self.optional_class_arguments))


class FactoryArgumentError(Exception):
    
    name: str

    def __init__(self, name: str) -> None:
        super(FactoryArgumentError, self).__init__()
        self.name = name


class MissingGlobalFactoryArgument(FactoryArgumentError):
    pass


class MissingFactoryArgument(FactoryArgumentError):
    pass


class InvalidFactoryArgument(FactoryArgumentError):
    pass
