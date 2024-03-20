"""Module for definition of useable deep learning models."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TypedDict


class Model(str, Enum):
    """Names of available CV models."""

    sku_gap = "sku_gap"
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


models: dict[Model, _ModelInfo] = {
    Model.yolov8_medium: {
        "info": "yxcv",
        "path": Path("weights/yolov8m.pt"),
    },
    Model.sku_gap: {
        "info": "yxcv",
        "path": Path("weights/yolov8_sku_gap.pt"),
    },
}
