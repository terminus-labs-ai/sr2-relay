from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from sr2_relay.models.canonical import CanonicalRequest

_registry: dict[str, "RequestTranslator"] = {}


class RequestTranslator(ABC):
    call_types: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "call_types"):
            instance = cls()
            for ct in cls.call_types:
                _registry[ct] = instance

    @abstractmethod
    def to_canonical(self, data: dict, *, call_type: str | None = None) -> CanonicalRequest: ...

    @abstractmethod
    def from_canonical(self, request: CanonicalRequest) -> dict: ...


def get_translator(call_type: str) -> RequestTranslator | None:
    return _registry.get(call_type)
