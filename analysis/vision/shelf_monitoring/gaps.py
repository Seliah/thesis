"""Module to implement the detection of gaps in an ongoing video stream."""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from logging import WARN, getLogger

from cv2.typing import MatLike
from reactivex import Observable, concat, repeat_value
from reactivex import operators as ops

from analysis.app_logging import logger
from analysis.definitions import PATH_SETTINGS
from analysis.types_adeck import settings
from analysis.vision.shelf_monitoring.models import Model, models

monitoring_settings = settings.load(PATH_SETTINGS).shelf_monitoring

N = 10
MEMORY_TIME = 20
"""How long to memorize gaps."""
TIME_PER_FRAME = 1
"""How long to wait between analyses."""
MEMORIZED_FRAME_COUNT = int(MEMORY_TIME / TIME_PER_FRAME)


def analyze_shelf(
    frames: Observable[MatLike],
    source_id: str,
    visualize: bool,
):
    """Analyze frames from given observable with shelf monitoring."""
    # Get warping bound points for this stream, if configured
    points = monitoring_settings.get(source_id, None)
    if points is None:
        return None
    logger.info(f'Starting shelf monitoring for "{source_id}"')

    from ultralytics import YOLO

    from analysis.util.image import show, warp
    from analysis.util.yolov8 import plot, predict
    from analysis.vision.shelf_monitoring.removal import has_new_gap

    # Prevent debug output from predictions
    getLogger("ultralytics").setLevel(WARN)

    model = YOLO(models[Model.sku_gap]["path"])

    def analyze_frame(image: MatLike):
        if points is not None:
            image = warp(image, points)
        results = predict(model, image, classes=[1])
        if visualize:
            show(plot(results[0], labels=True, line_width=1), fps=7)
        return results

    result_stream = frames.pipe(
        # Get analysis results for every n-th frame
        ops.buffer_with_count(N),
        # Return only the last frame
        ops.map(lambda images: images[-1]),
        ops.map(analyze_frame),
    )
    return concat(repeat_value(None, MEMORIZED_FRAME_COUNT), result_stream).pipe(
        # Emit all previous results as well
        ops.buffer_with_count(MEMORIZED_FRAME_COUNT - 1, 1),
        ops.map(has_new_gap),
    )


def parse_shelf_result(has_new: bool, source_id: str):
    """Print status message for given shelf analysis result."""
    if has_new:
        logger.info(f"{source_id} - Neue Entnahme")
