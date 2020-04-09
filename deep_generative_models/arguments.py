from typing import List

from deep_generative_models.configuration import Configuration


class ArgumentValidator:

    def mandatory_arguments(self) -> List[str]:
        return []

    def optional_arguments(self) -> List[str]:
        return []

    def validate_arguments(self, arguments: Configuration) -> None:
        # keep the remaining arguments here
        remaining_arguments = set(arguments.keys())

        # mandatory arguments
        for mandatory_argument in self.mandatory_arguments():
            if mandatory_argument in remaining_arguments:
                remaining_arguments.remove(mandatory_argument)
            else:
                raise MissingArgument(mandatory_argument)

        # optional arguments
        for optional_argument in self.optional_arguments():
            if optional_argument in remaining_arguments:
                remaining_arguments.remove(optional_argument)

        # invalid arguments
        for remaining_argument in remaining_arguments:
            raise InvalidArgument(remaining_argument)


class ArgumentError(Exception):
    name: str

    def __init__(self, name: str) -> None:
        super(ArgumentError, self).__init__()
        self.name = name


class MissingArgument(ArgumentError):
    pass


class InvalidArgument(ArgumentError):
    pass
