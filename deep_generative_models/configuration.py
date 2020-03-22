import json

from typing import Dict, Any, List, Optional


class Configuration:

    def __init__(self, values: Dict) -> None:
        self.values = {}
        for key, value in values.items():
            self.values[key] = self._transform_value(value)

    def __contains__(self, item: str) -> bool:
        return item in self.values

    def __getattr__(self, item: str) -> Any:
        # if a method in the dictionary is being called
        # WARNING: no config entries can be called like dictionary methods
        if hasattr(self.values, item):
            return getattr(self.values, item)
        # if not check for dictionary elements
        else:
            assert item in self, "Configuration entry '{}' not found.".format(item)
            return self.values[item]

    def __getitem__(self, item: str) -> Any:
        assert item in self, "Configuration entry '{}' not found.".format(item)
        return self.values[item]

    def _transform_value(self, value: Any) -> Any:
        if type(value) == dict:
            return Configuration(value)
        if type(value) == list:
            return [self._transform_value(child_value) for child_value in value]
        # I added the Configuration type here because it is easier for some border case uses
        elif type(value) in [str, int, float, Configuration]:
            return value
        else:
            raise Exception("Unexpected configuration value type '{}'.".format(str(type(value))))

    def get(self, item: str, default: Optional[Any] = None, transform_default: bool = True) -> Any:
        if item in self:
            return getattr(self, item)
        elif transform_default:
            return self._transform_value(default)
        else:
            return default

    def get_all_defined(self, items: List[str]) -> Dict[str, Any]:
        defined = {}
        for item in items:
            if item in self:
                defined[item] = getattr(self, item)
        return defined


def load_configuration(path: str) -> Configuration:
    with open(path, "r") as configuration_file:
        return Configuration(json.load(configuration_file))