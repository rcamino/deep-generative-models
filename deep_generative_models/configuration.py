import json

from typing import Dict, Any, List, Optional

from deep_generative_models.dictionary import Dictionary


class Configuration(Dictionary[Any]):

    @classmethod
    def _wrap_recursively(cls, value: Any) -> Any:
        # I added the Configuration type here because it is easier for some border case uses
        if value is None or type(value) in [str, int, float, bool, Configuration]:
            return value
        if type(value) == dict:
            return Configuration(value)
        if type(value) == list:
            return [cls._wrap_recursively(child_value) for child_value in value]
        else:
            raise Exception("Unexpected configuration value type '{}'.".format(str(type(value))))

    @classmethod
    def _unwrap_recursively(cls, value: Any) -> Any:
        if value is None or type(value) in [str, int, float, bool]:
            return value
        if type(value) == Configuration:
            unwrapped = {}
            for key, child_value in value.items():
                unwrapped[key] = cls._unwrap_recursively(child_value)
            return unwrapped
        if type(value) == list:
            return [cls._unwrap_recursively(child_value) for child_value in value]
        else:
            raise Exception("Unexpected configuration value type '{}'.".format(str(type(value))))

    def __init__(self, wrapped: Optional[Dict[str, Any]] = None) -> None:
        # don't send the wrapped values yet
        super(Configuration, self).__init__()
        # wrap and add recursively now that the dictionary is initialized
        if wrapped is not None:
            for name, value in wrapped.items():
                self[name] = self._wrap_recursively(value)

    def get(self, name: str, default: Optional[Any] = None, transform_default: bool = True,
            unwrap: bool = False) -> Any:
        if name in self:
            if unwrap:
                return self._unwrap_recursively(self[name])
            else:
                return self[name]
        elif transform_default:
            return self._wrap_recursively(default)
        else:
            return default

    def get_all_defined(self, names: List[str]) -> Dict[str, Any]:
        defined = {}
        for name in names:
            if name in self:
                defined[name] = self[name]
        return defined


def load_configuration(path: str) -> Configuration:
    with open(path, "r") as configuration_file:
        return Configuration(json.load(configuration_file))
