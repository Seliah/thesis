from threading import Event
from typing import Tuple

from cv2 import COLOR_BGR2GRAY, VideoCapture, absdiff, cvtColor, imshow, waitKey
from cv2.typing import MatLike
from reactivex import operators as ops

from motion.motion import show_four
from util.rx import from_capture


def analyze_image(images: Tuple[MatLike, MatLike]):
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
).subscribe(analyze_image)
