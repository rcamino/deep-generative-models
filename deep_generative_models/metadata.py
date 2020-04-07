import json

from typing import Dict, Generator, List


class VariableMetadata:

    metadata: Dict
    variable_index: int

    def __init__(self, metadata: Dict, variable_index: int) -> None:
        self.metadata = metadata
        self.variable_index = variable_index

    def get_index(self) -> int:
        return self.variable_index

    def get_feature_index(self) -> int:
        return sum(self.metadata["variable_sizes"][:self.get_index()])

    def get_type(self) -> str:
        return self.metadata["variable_types"][self.variable_index]

    def get_name(self) -> str:
        return self.metadata["variables"][self.variable_index]

    def get_size(self) -> int:
        return self.metadata["variable_sizes"][self.variable_index]

    def is_binary(self) -> bool:
        return self.get_type() == "binary"

    def is_categorical(self) -> bool:
        return self.get_type() == "categorical"

    def is_numerical(self) -> bool:
        return self.get_type() == "numerical"

    def get_absolute_index_from_value(self, value: str) -> int:
        if not self.is_categorical():
            raise Exception("The variable '{}' is not categorical.".format(self.get_name()))
        return self.metadata["value_to_index"][self.get_name()][value]

    def get_relative_index_from_value(self, value: str) -> int:
        if not self.is_categorical():
            raise Exception("The variable '{}' is not categorical.".format(self.get_name()))
        return self.get_absolute_index_from_value(value) - self.get_feature_index()

    def get_value_from_absolute_index(self, absolute_index: int) -> int:
        if not self.is_categorical():
            raise Exception("The variable '{}' is not categorical.".format(self.get_name()))
        name, value = self.metadata["index_to_value"][absolute_index]
        if name != self.get_name():
            raise Exception("The absolute index {:d} does not belong to the variable '{}'."
                            .format(absolute_index, self.get_name()))
        return value

    def get_value_from_relative_index(self, relative_index: int) -> int:
        if not self.is_categorical():
            raise Exception("The variable '{}' is not categorical.".format(self.get_name()))
        return self.get_value_from_absolute_index(relative_index + self.get_feature_index())

    def get_values(self) -> List[str]:
        if not self.is_categorical():
            raise Exception("The variable '{}' is not categorical.".format(self.get_name()))
        return self.metadata["value_to_index"][self.get_name()].keys()


class Metadata:

    metadata: Dict

    def __init__(self, metadata: Dict) -> None:
        self.metadata = metadata

    def get_by_variable(self) -> Generator[VariableMetadata, None, None]:
        for variable_index in range(len(self.metadata["variables"])):
            yield self.get_variable_by_index(variable_index)

    def get_variable_by_index(self, variable_index: int) -> VariableMetadata:
        return VariableMetadata(self.metadata, variable_index)

    def get_variable_by_name(self, variable_name: str) -> VariableMetadata:
        variable_index = self.metadata["variables"].index(variable_name)
        return VariableMetadata(self.metadata, variable_index)

    def get_num_variables(self) -> int:
        return len(self.metadata["variables"])

    def get_num_samples(self) -> int:
        return self.metadata["num_samples"]

    def get_num_features(self) -> int:
        return self.metadata["num_features"]


def load_metadata(path: str) -> Metadata:
    with open(path, "r") as configuration_file:
        return Metadata(json.load(configuration_file))
