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
from typing import List, Optional

from cv2 import WINDOW_NORMAL, VideoCapture, imread, imshow, namedWindow, waitKey
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
shelf_app = Typer(help="Run shelf monitoring.")
yolo_app = Typer(help="Use YOLOv8 with different models on images or videos for testing.")

monitoring_settings = settings.load(PATH_SETTINGS).shelf_monitoring


@yolo_app.command()
def info(model_id: Annotated[Model, Argument(help="Which model to use for analysis.")]):
    """Print out information about a given model."""
    from ultralytics import YOLO

    model = YOLO(models[model_id]["path"])
    console.print("Classes:")
    console.print(models[model_id]["info"])
    console.print(model.names)  # pyright: ignore[reportUnknownMemberType]


@yolo_app.command()
def image(  # noqa: PLR0913 we need complex parameters for cli execution
    model_id: Annotated[Model, Argument(help="Which model to use for analysis.")],
    path: Annotated[Path, Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, Option(help="Whether to show class names or not.")] = False,
    conf: Annotated[float, Option(help="Whether to show class names or not.")] = 0.25,
    classes: Annotated[Optional[List[int]], Option(help="The classes to look for.")] = None,
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

    results = predict(model, img, stream=False, classes=classes, conf=conf)
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


@shelf_app.command(name="stream")
def shelf_stream(
    source: Annotated[str, Argument(help="Video source URL for input stream.")],
    crop_like: Annotated[
        str,
        Option(help="Crop the image like the configuration of the given ID (see settings file)."),
    ],
):
    """Analyze a video stream."""
    capture = VideoCapture(source)
    logger.info("Starting")
    results = analyze_shelf(from_capture(capture, Event()), crop_like, visualize=True)
    if results is not None:
        results.subscribe(lambda result: parse_shelf_result(result, crop_like), logger.exception)


if __name__ == "__main__":
    shelf_app()
