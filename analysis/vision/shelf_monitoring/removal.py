"""Module for defining logic for detecting removal from shelves."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from analysis.app_logging import logger
from analysis.util.yolov8 import compare_results

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Results

OVERLAP_THRESHOLD = 0.8
NEEDED_OCCURENCES = 2


def has_new_gap(ossa_results: list[list[Results] | None]):
    """Check if the last result in the given list has any new gap detections."""
    newest = ossa_results[-1]
    if newest is None or newest[0].boxes is None:
        # There was no gap detected, there cant be a new one
        return False
    # Get non None previous results into a iterable, but reversed
    previous = (results for results in reversed(ossa_results[0:-1]) if results is not None)
    # Get box overlaps between the current results and each previous result
    overlaps = [
        compare_results(newest[0].boxes, previous_boxes)
        for previous_results in previous
        if (previous_boxes := previous_results[0].boxes) is not None
    ]
    current_boxes_bounds = newest[0].boxes.xyxy  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # Check if any of the boxes is a new gap
    return any(_is_new_gap(index, overlaps) for index, _bounds in enumerate(current_boxes_bounds))  # pyright: ignore[reportUnknownArgumentType]


def _is_new_gap(index: int, all_overlaps: Iterable[Tensor]):
    occurences = 0
    for overlaps in all_overlaps:
        for overlap in overlaps[index]:
            if overlap > OVERLAP_THRESHOLD:
                occurences += 1
                if occurences > NEEDED_OCCURENCES:
                    # Multiple overlapping gaps found, this one is not new
                    if __debug__:
                        logger.debug(f"Found previously detected gaps for gap at {index}. This one is not new.")
                    return False
    if occurences == NEEDED_OCCURENCES:
        # Only one overlapping gap was found, this one is new
        if __debug__:
            logger.debug("Gap is new!")
        return True
    # Not enough previously overlapping gaps found, this one is probably a false positive
    if __debug__:
        logger.debug(
            f"Not enough previously overlapping gaps found for gap at {index}. This one is probably a false positive.",
        )
    return False
