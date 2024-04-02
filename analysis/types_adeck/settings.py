"""Module that defines the type for program settings."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import rich
import tomllib
from pydantic.dataclasses import dataclass

from analysis.definitions import PATH_SETTINGS
from analysis.types_adeck import BaseType, parse_with_raise
from analysis.util.image import (
    RectPoints,  # noqa: TCH001 this can't be in type checking block - taht results in a pydantic error
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Settings(BaseType):
    """Dataclass for cameras."""

    excludes: list[str]
    shelf_monitoring: dict[str, RectPoints]

    @staticmethod
    def repr_raw(json: Mapping[str, Any]) -> str:  # noqa: ARG004
        """Return a lossy string representation for the data of this settings definition.

        TODO(elias): use str_raw instead of this.
        """
        return "program settings"


def load(path: Path):
    """Load and parse settings from the given toml file path."""
    with path.open("rb") as file:
        data = tomllib.load(file)
        return parse_with_raise(Settings, data)


if __name__ == "__main__":
    rich.print(load(PATH_SETTINGS))
