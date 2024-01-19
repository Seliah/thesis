from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, List, Optional, Type, TypeVar

from pydantic import ValidationError
from pydantic.dataclasses import dataclass

_logger = getLogger(__name__)


@dataclass
class BaseType(ABC):
    """Base Type for high level dataclass types defined in this application."""

    @staticmethod
    @abstractmethod
    def repr_raw(json: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def __repr__(json: Any) -> str:
        raise NotImplementedError


T = TypeVar("T", bound=BaseType)


def parse(app_type: Type[T], json: Any) -> Optional[T]:
    """Parse a raw json dict into a object of the given type.

    :return: None - Validation error occurred.
    """
    try:
        return app_type(**json)
    except ValidationError as exception:
        _logger.error(
            f"Error parsing camera data for {app_type.repr_raw(json)}: {exception}",
        )


# Type annotation is needed here as language server is confused
def parse_all(app_type: Type[T], json: Any) -> List[T]:
    """Parse a raw json array with dict entries into an array with entries of the given type.

    This filters out entries that had a validation error.
    """
    return [parsed for entry in json if (parsed := parse(app_type, entry)) is not None]
