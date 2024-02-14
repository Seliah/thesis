"""Module for expanding or typing YOLOv8 functionality."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cv2.typing import MatLike, Rect
from numpy.typing import NDArray
from torch import Tensor
from torchvision.ops import box_iou
from ultralytics.engine.results import Boxes, Results

if TYPE_CHECKING:
    from ultralytics import YOLO

Source = str | int | NDArray[Any] | Path


def predict(model: YOLO, source: Source, stream: bool = False, classes: list[int] | None = None, conf: float = 0.25):
    """YOLOv8 prediction (but typed).

    See https://docs.ultralytics.com/modes/predict/#inference-arguments
    """
    return cast(list[Results], model.predict(source, stream, classes=classes, conf=conf))  # pyright: ignore[reportUnknownMemberType]


def plot(results: Results, labels: bool = True, line_width: float | None = None):
    """Plot results (but typed).

    See https://docs.ultralytics.com/reference/engine/results/?h=plot#ultralytics.engine.results.Results.plot
    """
    return cast(MatLike, results.plot(labels=labels, line_width=line_width))  # pyright: ignore[reportUnknownMemberType]


def compare_results(boxes1: Boxes, boxes2: Boxes):
    """Get overlaps from given result box lists."""
    return box_iou(cast(Tensor, boxes1.xyxy), cast(Tensor, boxes2.xyxy))  # pyright: ignore[reportUnknownMemberType]


def get_rect_from_box(box: Tensor) -> Rect:
    """Get cv2 Rect from given result box."""
    bounds = cast(list[float], box.tolist())  # pyright: ignore[reportUnknownMemberType]
    return _get_rect_from_bounds(bounds)


def _get_rect_from_bounds(bounds: list[float]) -> Rect:
    coords = (bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1])
    return [int(coord) for coord in coords]
