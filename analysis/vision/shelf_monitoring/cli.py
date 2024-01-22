"""Module for definition of motion_search debug cli.

See __main__ for application.

You will see that import are done in many commands.
These imports take a while and are therefore "lazy loaded" like this.
They are only imported, when they are actually needed inside a command.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from pathlib import Path
from threading import Event
from typing import List, Optional, cast

import typer
from cv2 import VideoCapture, imread, imshow, rectangle, resize, waitKey
from cv2.typing import MatLike, Rect
from reactivex import operators as ops
from reactivex.operators import map as map_op
from reactivex.operators import throttle_first
from rich.console import Console
from typing_extensions import Annotated

from analysis.util.image import warp
from analysis.util.rx import from_capture
from analysis.vision.shelf_monitoring.models import Model, models

console = Console()
app = typer.Typer()


class _AnalysisError(Exception):
    """Error to raise when a problem happened during analysis of an image."""


@app.command()
def info(model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")]):
    """Print out heatmap data for a given camera."""
    from ultralytics import YOLO

    model = YOLO(models[model_id]["path"])
    console.print("Classes:")
    console.print(models[model_id]["info"])
    console.print(model.names)


@app.command()
def image(
    model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")],
    path: Annotated[Path, typer.Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, typer.Option(help="Whether or not to show class names.")] = False,
    conf: Annotated[Optional[float], typer.Option(help="Whether or not to show class names.")] = None,
):
    """Analyze a given image."""
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

    model = YOLO(models[model_id]["path"])
    results = cast(List[Results], model.predict(path, conf=conf))
    for result in results:
        imshow("Results", result.plot(labels=labels, line_width=2))
        waitKey(0)


@app.command()
def stream(
    source: Annotated[str, typer.Argument(help="Video source URL for input stream.")],
    model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")],
    labels: Annotated[bool, typer.Argument(help="Whether or not to show class names.")] = False,
):
    """Analyze a given video stream."""
    from ultralytics import YOLO

    model = YOLO(models[model_id]["path"])
    # for result in model.predict(source, stream=True):
    for result in model.predict(source, stream=True, classes=[0]):
        plotted = result.plot(labels=labels, line_width=2)
        imshow("Results", resize(plotted, (1440, 810)))
        waitKey(1)


@app.command()
def shelf(
    path_img1: Annotated[Path, typer.Argument(help="The path of the older image.")],
    path_img2: Annotated[Path, typer.Argument(help="The path of the newer image.")],
    overlap_threshold: Annotated[float, typer.Option(help="How big the detection box overlap must be (0-1).")] = 0.1,
):
    """Analyze two images with shelf monitoring.

    This will detect missing items in img2 that were present in img1.
    """
    from torchvision.ops import box_iou
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

    model_sku = YOLO(models[Model.sku]["path"])
    model_ossa = YOLO(models[Model.ossa]["path"])

    result_sku = cast(List[Results], model_sku.predict(path_img1))[0]
    sku_boxes = result_sku.boxes
    if sku_boxes is None:
        raise _AnalysisError("Analysis for products failed.")

    result_ossa = cast(List[Results], model_ossa.predict(path_img2))[0]
    ossa_boxes = result_ossa.boxes
    if ossa_boxes is None:
        raise _AnalysisError("Analysis for gaps failed.")

    overlapping = box_iou(result_ossa.boxes.xyxy, result_sku.boxes.xyxy)
    for gap_index, item_index in overlapping.nonzero():
        overlap = overlapping[gap_index, item_index]
        if overlap > overlap_threshold:
            box = cast(List[float], result_sku.boxes.xyxy[item_index].tolist())
            rectangle(result_ossa.orig_img, _get_rect_from_box(box), (0, 255, 0), 2)

    imshow("Results", result_ossa.orig_img)
    waitKey(0)


def _get_rect_from_box(box: List[float]) -> Rect:
    coords = (box[0], box[1], box[2] - box[0], box[3] - box[1])
    return [int(coord) for coord in coords]


@app.command()
def cropped(
    path: Annotated[Path, typer.Argument(help="The path of the to be cropped image.")],
):
    """Show the given image in a cropped representation, which will be used for analysis."""
    img = imread(str(path))
    # 3
    points = ((780, 160), (1840, 490), (1580, 930), (800, 630))
    # 2
    # points = ((1060, 350), (1790, 590), (1580, 930), (1020, 720))  # noqa: ERA001
    image_warped = warp(img, points)
    imshow("Cropped", image_warped)
    waitKey(0)


@app.command()
def shelf_stream(
    source: Annotated[str, typer.Argument(help="Video source URL for input stream.")],
):
    """Analyze a video stream with shelf monitoring."""
    from ultralytics import YOLO

    points = ((780, 160), (1840, 490), (1580, 930), (800, 630))
    cap = VideoCapture(source)
    model_sku = YOLO(models[Model.sku]["path"])
    model_ossa = YOLO(models[Model.ossa]["path"])

    from_capture(cap, Event()).pipe(
        throttle_first(1 / 2),
        map_op(lambda image: warp(image, points)),
        ops.buffer_with_count(5),
        # ops.do_action(lambda values: console.print(values[-1])),
        # map_op(model_sku.predict),
        # pairwise(),
    ).subscribe()
    # ).subscribe(lambda images: show(cap, images[1][0].plot()))


def _show(cap: VideoCapture, image: MatLike):
    imshow("Video", image)
    if waitKey(1) == ord("q"):
        cap.release()


if __name__ == "__main__":
    app()
