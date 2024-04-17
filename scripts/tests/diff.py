"""Module for trying out the concept for motion detection."""
from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING

from cv2 import COLOR_BGR2GRAY, VideoCapture, absdiff, cvtColor, imshow, waitKey
from reactivex import operators as ops

from analysis.app_logging import logger
from analysis.util.rx import from_capture
from analysis.vision.motion_search.motion import show_four

if TYPE_CHECKING:
    from cv2.typing import MatLike

def _analyze_image(images: tuple[MatLike, MatLike]):
    four = show_four(
        images[0],
        images[1],
        absdiff(images[0], images[1]),
        images[1],
    )
    imshow("Difference", four)
    waitKey(1)


from_capture(VideoCapture(0), Event()).pipe(
    ops.throttle_first(2),
    ops.map(lambda image: cvtColor(image, COLOR_BGR2GRAY)),
    ops.pairwise(),
).subscribe(_analyze_image, logger.exception)
