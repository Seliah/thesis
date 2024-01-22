"""Module for expanding the functionality of RxPY."""
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Event
from typing import Optional

import cv2
from cv2.typing import MatLike
from reactivex import Observable, create
from reactivex.abc import ObserverBase, SchedulerBase
from reactivex.disposable import Disposable

_logger = getLogger(__name__)

executor = ThreadPoolExecutor(None, "Capture")


def from_capture(capture: cv2.VideoCapture, termination_event: Event) -> Observable[MatLike]:
    """Create an observable from an opencv capture."""

    def on_subscribe(observer: ObserverBase[MatLike], _: Optional[SchedulerBase]):
        disposed = Event()
        while capture.isOpened() and not termination_event.is_set() and not disposed.is_set():
            success, frame = capture.read()
            if not success:
                _logger.error("Observable OpenCV Capture was not successful.")
                break
            observer.on_next(frame)
        capture.release()
        observer.on_completed()

        def dispose():
            disposed.set()

        return Disposable(dispose)

    return create(on_subscribe)
