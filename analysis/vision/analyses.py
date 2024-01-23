"""Module for defining analyses.

This provides a central place for all analysis definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from analysis.vision.motion_search.motion import analyze_motion, update_global_matrix, write_motion

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from reactivex import Observable
    from reactivex.subject import Subject

T = TypeVar("T")


@dataclass
class Analysis(Generic[T]):
    """Class that describes a analysis definition.

    That definition formulates how video feeds should be analyzed with callbacks.
    """

    analyze: Callable[[Subject[MatLike], bool], Observable[T]]
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
}
