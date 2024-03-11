"""Module for defining analyses.

This provides a central place for all analysis definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from analysis.vision.motion_search.motion import analyze_motion, update_global_matrix, write_motion
from analysis.vision.shelf_monitoring.cli import analyze_shelf, parse_shelf_result

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from reactivex import Observable

T = TypeVar("T")


@dataclass
class Analysis(Generic[T]):
    """Class that describes a analysis definition.

    That definition formulates how video feeds should be analyzed with callbacks.
    """

    analyze: Callable[[Observable[MatLike], str, bool], Observable[T] | None]
    """Analyze the given frame observable in some way.

    This logic will be run in a child process.
    Analysis results will be streamed to the main thread to be parsed.
    """
    parse: Callable[[T, str], None]
    """Parse the results of the analysis done by `analyze`."""
    on_termination: Callable[[], None] | None = None
    """Do something before the program shuts down, for example saving to disk."""


analyses = {
    "motion_search": Analysis(analyze_motion, update_global_matrix, write_motion),
    "shelf_monitoring": Analysis(analyze_shelf, parse_shelf_result),
}
"""Dictionary for the definition of to be done analyses.

Every analysis here will be applied to each camera.
"""
