"""Module that defines the type for program settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import rich
import tomllib
from pydantic.dataclasses import dataclass

from analysis.types_adeck import BaseType, parse_with_raise
from analysis.util.image import RectPoints

DEFAULT_PATH = Path("./settings.toml")


@dataclass
class Settings(BaseType):
    """Dataclass for cameras."""

    excludes: list[str]
    shelf_monitoring: dict[str, RectPoints]

    @staticmethod
    def repr_raw(json: Mapping[str, Any]) -> str:
        """Return a lossy string representation for the data of this settings definition.

        TODO(elias): use str_raw instead of this.
        """
        return "program settings"


def load(path: Path):
    with path.open("rb") as file:
        data = tomllib.load(file)
        return parse_with_raise(Settings, data)


if __name__ == "__main__":
    rich.print(load(DEFAULT_PATH))
