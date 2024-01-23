from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from cv2.typing import MatLike
from reactivex import Observable
from reactivex.subject import Subject

from analysis.vision.motion_search.motion import analyze_motion, update_global_matrix, write_motion

T = TypeVar("T")


@dataclass
class Analysis(Generic[T]):
    analyze: Callable[[Subject[MatLike], bool], Observable[T]]
    parse: Callable[[T, str], None]
    on_termination: Callable[[], None] | None = None


analyses = {
    "motion_search": Analysis(analyze_motion, update_global_matrix, write_motion),
}
