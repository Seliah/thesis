"""Module for definition of shelf monitoring debug cli.

See __main__ for application.

You will see that import are done in many commands.
These imports take a while and are therefore "lazy loaded" like this.
They are only imported, when they are actually needed inside a command.

The prdictions may be improved further by tweaking args like conf, iou or agnostic_nms.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from logging import WARN, getLogger
from pathlib import Path
from threading import Event
from typing import Optional

from cv2 import CAP_PROP_FPS, WINDOW_NORMAL, VideoCapture, imread, imshow, namedWindow, rectangle, waitKey
from cv2.typing import MatLike
from reactivex import concat, repeat_value
from reactivex import operators as ops
from rich.console import Console
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from analysis.app_logging import logger
from analysis.types_adeck import settings
from analysis.util.image import show, warp
from analysis.util.rx import from_capture
from analysis.vision.shelf_monitoring.models import Model, models

console = Console()
shelf_app = Typer()
yolo_app = Typer()

N = 10

class _AnalysisError(Exception):
    """Error to raise when a problem happened during analysis of an image."""


@yolo_app.command()
def info(model_id: Annotated[Model, Argument(help="Which model to use for analysis.")]):
    """Print out heatmap data for a given camera."""
    from ultralytics import YOLO

    model = YOLO(models[model_id]["path"])
    console.print("Classes:")
    console.print(models[model_id]["info"])
    console.print(model.names)  # pyright: ignore[reportUnknownMemberType]


@yolo_app.command()
def image(
    model_id: Annotated[Model, Argument(help="Which model to use for analysis.")],
    path: Annotated[Path, Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, Option(help="Whether to show class names or not.")] = False,
    conf: Annotated[float, Option(help="Whether to show class names or not.")] = 0.25,
    crop_like: Annotated[Optional[str], Option(help="Crop the image like the configuration of the given ID.")] = None,
    settings_path: Annotated[
        Path,
        Option(help="Path to the analysis settings file - needed when crop_like is given."),
    ] = settings.DEFAULT_PATH,
):
    """Analyze a given image."""
    from ultralytics import YOLO

    from analysis.util.yolov8 import plot, predict

    model = YOLO(models[model_id]["path"])
    img = imread(str(path))

    if crop_like is not None:
        monitoring_settings = settings.load(settings_path).shelf_monitoring
        points = monitoring_settings.get(crop_like, None)
        if points is None:
            logger.error(f'No points configuration found for "{crop_like}"')
            return
        img = warp(img, points)

    results = predict(model, img, conf=conf)
    namedWindow("Results", WINDOW_NORMAL)
    for result in results:
        imshow("Results", plot(result, labels=labels, line_width=2))
        waitKey(0)


@yolo_app.command()
def stream(
    model_id: Annotated[Model, Argument(help="Which model to use for analysis.")],
    source: Annotated[str, Argument(help="Video source URL for input stream.")],
    labels: Annotated[bool, Argument(help="Whether or not to show class names.")] = False,
):
    """Analyze a given video stream."""
    from ultralytics import YOLO

    from analysis.util.yolov8 import plot, predict

    model = YOLO(models[model_id]["path"])
    namedWindow("Results", WINDOW_NORMAL)
    for result in predict(model, source, stream=True):
        imshow("Results", plot(result, labels=labels, line_width=2))
        waitKey(1)


@shelf_app.command()
def diff(
    path_img1: Annotated[Path, Argument(help="The path of the older image.")],
    path_img2: Annotated[Path, Argument(help="The path of the newer image.")],
    overlap_threshold: Annotated[float, Option(help="How big the detection box overlap must be (0-1).")] = 0.1,
):
    """Analyze two images with shelf monitoring.

    This will detect missing items in img2 that were present in img1.
    """
    from ultralytics import YOLO

    from analysis.util.yolov8 import compare_results, get_rect_from_box, predict

    model_sku = YOLO(models[Model.sku]["path"])
    model_ossa = YOLO(models[Model.ossa]["path"])

    result_sku = predict(model_sku, path_img1)[0]
    sku_boxes = result_sku.boxes
    if sku_boxes is None:
        raise _AnalysisError("Analysis for products failed.")

    result_ossa = predict(model_ossa, path_img2)[0]
    ossa_boxes = result_ossa.boxes
    if ossa_boxes is None:
        raise _AnalysisError("Analysis for gaps failed.")

    overlapping = compare_results(ossa_boxes, sku_boxes)
    for gap_index, item_index in overlapping.nonzero():
        overlap = overlapping[gap_index, item_index]
        if overlap > overlap_threshold:
            box = sku_boxes.xyxy[item_index]  # pyright: ignore[reportUnknownMemberType]
            rectangle(result_ossa.orig_img, get_rect_from_box(box), (0, 255, 0), 2)

    imshow("Results", result_ossa.orig_img)
    waitKey(0)


@shelf_app.command(name="stream")
def shelf_stream(
    source: Annotated[str, Argument(help="Video source URL for input stream.")],
    memory_time: Annotated[float, Option(help="How long to memorize gaps.")] = 20,
    time_per_frame: Annotated[float, Option(help="How long to wait between analyses.")] = 1,
    crop_like: Annotated[Optional[str], Option(help="Crop the image like the configuration of the given ID.")] = None,
    settings_path: Annotated[
        Path,
        Option(help="Path to the analysis settings file - needed when crop_like is given."),
    ] = settings.DEFAULT_PATH,
):
    """Analyze a video stream with shelf monitoring."""
    from ultralytics import YOLO

    from analysis.util.yolov8 import plot, predict
    from analysis.vision.shelf_monitoring.removal import has_new_gap

    memorized_frame_count = int(memory_time / time_per_frame)

    cap = VideoCapture(source)
    fps = round(cap.get(CAP_PROP_FPS))
    logger.info(f"FPS: ~{fps}")

    model = YOLO(models[Model.sku_gap]["path"])

    # Get warping bound points for this stream, if configured
    if crop_like is not None:
        monitoring_settings = settings.load(settings_path).shelf_monitoring
        points = monitoring_settings.get(crop_like, None)
        if points is None:
            logger.error(f'No points configuration found for "{crop_like}"')
            return
    else:
        points = None

    def analyze_shelf(image: MatLike):
        if points is not None:
            image = warp(image, points)
        results = predict(model, image, classes=[1])
        show(cap, plot(results[0], labels=True, line_width=1), int(fps / 4))
        return results

    logger.info("Starting")
    logger.info(f"Memorizing {memorized_frame_count} frames.")
    # Prevent debug output from predictions
    getLogger("ultralytics").setLevel(WARN)
    # Get analysis results for every n-th frame
    results = from_capture(cap, Event()).pipe(
        ops.buffer_with_count(N),
        # Return only the last frame
        ops.map(lambda images: images[-1]),
        ops.map(analyze_shelf),
    )
    concat(repeat_value(None, memorized_frame_count), results).pipe(
        # Emit all previous results as well
        ops.buffer_with_count(memorized_frame_count - 1, 1),
        ops.map(has_new_gap),
    ).subscribe(lambda has_new: logger.info("Neue Entnahme") if has_new else None, logger.exception)


if __name__ == "__main__":
    shelf_app()
