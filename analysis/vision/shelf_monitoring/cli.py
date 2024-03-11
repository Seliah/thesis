"""Module for definition of shelf monitoring debug cli.

See __main__ for application.

You will see that import are done in many commands.
These imports take a while and are therefore "lazy loaded" like this.
They are only imported, when they are actually needed inside a command.

The prdictions may be improved further by tweaking args like conf, iou or agnostic_nms.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from pathlib import Path
from threading import Event
from typing import Optional

from cv2 import WINDOW_NORMAL, VideoCapture, imread, imshow, namedWindow, rectangle, waitKey
from rich.console import Console
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from analysis.app_logging import logger
from analysis.definitions import PATH_SETTINGS
from analysis.types_adeck import settings
from analysis.util.image import warp
from analysis.util.rx import from_capture
from analysis.vision.shelf_monitoring.gaps import analyze_shelf, parse_shelf_result
from analysis.vision.shelf_monitoring.models import Model, models

console = Console()
shelf_app = Typer()
yolo_app = Typer()

monitoring_settings = settings.load(PATH_SETTINGS).shelf_monitoring


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
):
    """Analyze a given image."""
    from ultralytics import YOLO

    from analysis.util.yolov8 import plot, predict

    model = YOLO(models[model_id]["path"])
    img = imread(str(path))

    if crop_like is not None:
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
    crop_like: Annotated[
        str,
        Option(help="Crop the image like the configuration of the given ID (see settings file)."),
    ],
):
    """Analyze a video stream with shelf monitoring."""
    capture = VideoCapture(source)
    logger.info("Starting")
    results = analyze_shelf(from_capture(capture, Event()), crop_like, visualize=True)
    if results is not None:
        results.subscribe(lambda result: parse_shelf_result(result, crop_like), logger.exception)


if __name__ == "__main__":
    shelf_app()
