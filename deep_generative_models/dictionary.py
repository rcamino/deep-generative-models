from typing import TypeVar, Generic, Dict, Optional, KeysView, ValuesView, ItemsView


T = TypeVar('T')


class Dictionary(Generic[T]):
    _wrapped: Dict[str, T]

    def __init__(self, wrapped: Optional[Dict[str, T]] = None) -> None:
        # avoid set item
        self.__dict__["_wrapped"] = {}
        # now can use set item
        if wrapped is not None:
            for key, value in wrapped.items():
                self[key] = value

    def __len__(self) -> int:
        return len(self._wrapped)

    def __contains__(self, key: str) -> bool:
        return key in self._wrapped

    def __getattr__(self, key: str) -> T:
        # forward to get item
        return self[key]

    def __setattr__(self, key: str, value: T) -> None:
        # forward to set item
        self[key] = value

    def __getitem__(self, key: str) -> T:
        return self._wrapped[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._wrapped[key] = value

    def keys(self) -> KeysView[str]:
        return self._wrapped.keys()

    def values(self) -> ValuesView[T]:
        return self._wrapped.values()

    def items(self) -> ItemsView[str, T]:
        return self._wrapped.items()
