"""Module for defining logic for detecting removal from shelves."""
from __future__ import annotations

from typing import TYPE_CHECKING

import rich

from analysis.util.yolov8 import compare_results

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Results

OVERLAP_THRESHOLD = 0.1


def has_new_gap(ossa_results: list[list[Results] | None]):
    """Check if the last result in the given list has any new gap detections."""
    newest = ossa_results[-1]
    if newest is None or newest[0].boxes is None:
        # There was no gap detected, there cant be a new one
        return False
    previous = [results for results in reversed(ossa_results[0:-1]) if results is not None]
    matches = [
        compare_results(newest[0].boxes, previous_boxes)
        for previous_results in previous
        if (previous_boxes := previous_results[0].boxes) is not None
    ]
    xyxy = newest[0].boxes.xyxy  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return any(_is_new_gap(index, matches) for index, _bounds in enumerate(xyxy))  # pyright: ignore[reportUnknownArgumentType]


def _is_new_gap(index: int, matches: list[Tensor]):
    for overlaps in matches:
        o = overlaps[index]
        if any(overlap for overlap in o if overlap > OVERLAP_THRESHOLD):
            # Overlapping gap was found, this one is not new
            rich.print(f"Found previously detected gap for gap at {index}.")
            return False
    # No overlapping gap was found, this one is new
    rich.print("Gap is new!")
    return True
