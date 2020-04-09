import json

from typing import Dict, Generator, List


class VariableMetadata:
    metadata: Dict

    def __init__(self, metadata: Dict) -> None:
        self.metadata = metadata

    def get_type(self) -> str:
        raise NotImplementedError

    def get_size(self) -> int:
        raise NotImplementedError

    def is_binary(self) -> bool:
        raise NotImplementedError

    def is_categorical(self) -> bool:
        raise NotImplementedError

    def is_numerical(self) -> bool:
        raise NotImplementedError

    def get_index_from_value(self, value: str) -> int:
        raise NotImplementedError

    def get_value_from_index(self, index: int) -> int:
        raise NotImplementedError

    def get_values(self) -> List[str]:
        raise NotImplementedError


class IndependentVariableMetadata(VariableMetadata):
    variable_index: int

    def __init__(self, metadata: Dict, variable_index: int) -> None:
        super(IndependentVariableMetadata, self).__init__(metadata)
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

    def get_feature_index_from_value(self, value: str) -> int:
        if not self.is_categorical():
            raise Exception("The independent variable '{}' is not categorical.".format(self.get_name()))
        return self.metadata["value_to_index"][self.get_name()][value]

    def get_index_from_value(self, value: str) -> int:
        if not self.is_categorical():
            raise Exception("The independent variable '{}' is not categorical.".format(self.get_name()))
        return self.get_feature_index_from_value(value) - self.get_feature_index()

    def get_value_from_feature_index(self, feature_index: int) -> int:
        if not self.is_categorical():
            raise Exception("The independent variable '{}' is not categorical.".format(self.get_name()))
        name, value = self.metadata["index_to_value"][feature_index]
        if name != self.get_name():
            raise Exception("The absolute index {:d} does not belong to the variable '{}'."
                            .format(feature_index, self.get_name()))
        return value

    def get_value_from_index(self, index: int) -> int:
        if not self.is_categorical():
            raise Exception("The independent variable '{}' is not categorical.".format(self.get_name()))
        return self.get_value_from_feature_index(index + self.get_feature_index())

    def get_values(self) -> List[str]:
        if not self.is_categorical():
            raise Exception("The independent variable '{}' is not categorical.".format(self.get_name()))
        return self.metadata["value_to_index"][self.get_name()].keys()


class DependentVariableMetadata(VariableMetadata):

    def get_type(self) -> str:
        # assuming the type from the size
        if self.is_binary():
            return "binary"
        elif self.is_categorical():
            return "categorical"
        elif self.is_numerical():
            return "numerical"
        else:
            raise Exception("Invalid response variable type.")

    def get_size(self) -> int:
        # assuming that if it has no classes it is numerical
        return self.metadata["num_classes"] if "num_classes" in self.metadata else 1

    def is_binary(self) -> bool:
        return self.get_size() == 2

    def is_categorical(self) -> bool:
        return self.get_size() > 2

    def is_numerical(self) -> bool:
        return self.get_size() == 1

    def get_index_from_value(self, value: str) -> int:
        if not self.is_categorical():
            raise Exception("The dependent variable is not categorical.")
        # WARNING: using linear search
        # pre-compute a dictionary if performance needed
        return self.metadata["classes"].index(value)

    def get_value_from_index(self, index: int) -> int:
        if not self.is_categorical():
            raise Exception("The dependent variable is not categorical.")
        return self.metadata["classes"][index]

    def get_values(self) -> List[str]:
        if not self.is_categorical():
            raise Exception("The dependent variable is not categorical.")
        return self.metadata["classes"]


class Metadata:
    metadata: Dict

    def __init__(self, metadata: Dict) -> None:
        self.metadata = metadata

    def get_by_independent_variable(self) -> Generator[IndependentVariableMetadata, None, None]:
        for variable_index in range(len(self.metadata["variables"])):
            yield self.get_independent_variable_by_index(variable_index)

    def get_independent_variable_by_index(self, variable_index: int) -> IndependentVariableMetadata:
        return IndependentVariableMetadata(self.metadata, variable_index)

    def get_independent_variable_by_name(self, variable_name: str) -> IndependentVariableMetadata:
        variable_index = self.metadata["variables"].index(variable_name)
        return IndependentVariableMetadata(self.metadata, variable_index)

    def get_dependent_variable(self) -> DependentVariableMetadata:
        return DependentVariableMetadata(self.metadata)

    def get_num_independent_variables(self) -> int:
        return self.metadata["num_variables"]

    def get_num_samples(self) -> int:
        return self.metadata["num_samples"]

    def get_num_features(self) -> int:
        return self.metadata["num_features"]


def load_metadata(path: str) -> Metadata:
    with open(path, "r") as configuration_file:
        return Metadata(json.load(configuration_file))
