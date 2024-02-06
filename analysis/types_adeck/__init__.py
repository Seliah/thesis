"""Module for implementing type definitions for data from the adeck systems vms.

These types can be used for automatic validation, type checking, linting and intellisense in IDEs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar

from pydantic import ValidationError
from pydantic.dataclasses import dataclass

from analysis.app_logging import logger


@dataclass
class BaseType(ABC):
    """Base Type for high level dataclass types defined in this application."""

    @staticmethod
    @abstractmethod
    def repr_raw(json: Mapping[str, Any]) -> str:
        """Return a lossless string representation of the data for this object."""
        raise NotImplementedError


T = TypeVar("T", bound=BaseType)


def parse_with_raise(app_type: type[T], json: Mapping[str, Any]):
    """Parse a raw json dict into a object of the given type and raise on ValidationError."""
    return app_type(**json)


def parse(app_type: type[T], json: Mapping[str, Any]):
    """Parse a raw json dict into a object of the given type.

    :return: None - Validation error occurred.
    """
    try:
        return app_type(**json)
    except ValidationError as exception:
        logger.error(
            f"Error parsing data for {app_type.repr_raw(json)}: {exception}",
        )


# Type annotation is needed here as language server is confused
def parse_all(app_type: type[T], json: list[Mapping[str, Any]]) -> list[T]:
    """Parse a raw json array with dict entries into an array with entries of the given type.

    This filters out entries that had a validation error.
    """
    return [parsed for entry in json if (parsed := parse(app_type, entry)) is not None]
