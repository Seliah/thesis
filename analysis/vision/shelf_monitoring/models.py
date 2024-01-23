"""Module for definition of useable deep learning models."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TypedDict


class Model(str, Enum):
    """Names of available cv models."""

    ossa = "ossa"
    sku = "sku110k"
    yolov8_nano = "n"
    yolov8_small = "s"
    yolov8_medium = "m"
    yolov8_large = "l"
    yolov8_x_large = "x"


class _ModelInfo(TypedDict):
    info: str
    """A description of the given model."""
    path: Path
    """The path of the weights file."""


models: dict[str, _ModelInfo] = {
    Model.ossa: {
        "info": "qwer",
        "path": Path("weights/yolov8_ossa.pt"),
    },
    Model.sku: {
        "info": "asdf",
        "path": Path("weights/yolov8_sku110k.pt"),
    },
    Model.yolov8_medium: {
        "info": "yxcv",
        "path": Path("weights/yolov8m.pt"),
    },
}
