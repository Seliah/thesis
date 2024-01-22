"""Module that defines data for deep learning based computer vision."""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Dict, List, Optional, TypedDict, cast

import typer
from cv2 import VideoCapture, imread, imshow, rectangle, resize, waitKey
from cv2.typing import MatLike, Rect
from reactivex import operators as ops
from reactivex.operators import map as map_op
from reactivex.operators import throttle_first
from rich.console import Console
from torchvision.ops import box_iou
from typing_extensions import Annotated
from ultralytics import YOLO
from ultralytics.engine.results import Results

from analysis.util.image import warp
from analysis.util.rx import from_capture

console = Console()
app = typer.Typer()


class _Model(str, Enum):
    ossa = "ossa"
    sku = "sku110k"
    m = "m"


class _ModelInfo(TypedDict):
    info: str
    """A description of the given model."""
    path: Path
    """The path of the weights file."""


models: Dict[str, _ModelInfo] = {
    _Model.ossa: {
        "info": "qwer",
        "path": Path("weights/yolov8_ossa.pt"),
    },
    _Model.sku: {
        "info": "asdf",
        "path": Path("weights/yolov8_sku110k.pt"),
    },
    _Model.m: {
        "info": "yxcv",
        "path": Path("weights/yolov8m.pt"),
    },
}


class _AnalysisError(Exception):
    """Error to raise when a problem happened during analysis of an image."""


@app.command()
def info(model_id: Annotated[_Model, typer.Argument(help="Which model to use for analysis.")]):
    """Print out heatmap data for a given camera."""
    model = YOLO(models[model_id]["path"])
    console.print("Classes:")
    console.print(models[model_id]["info"])
    console.print(model.names)


@app.command()
def image(
    model_id: Annotated[_Model, typer.Argument(help="Which model to use for analysis.")],
    path: Annotated[Path, typer.Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, typer.Option(help="Whether or not to show class names.")] = False,
    conf: Annotated[Optional[float], typer.Option(help="Whether or not to show class names.")] = None,
):
    """Analyze a given image."""
    model = YOLO(models[model_id]["path"])
    results = cast(List[Results], model.predict(path, conf=conf))
    for result in results:
        imshow("Results", result.plot(labels=labels, line_width=2))
        waitKey(0)


@app.command()
def stream(
    source: Annotated[str, typer.Argument(help="Video source URL for input stream.")],
    model_id: Annotated[_Model, typer.Argument(help="Which model to use for analysis.")],
    labels: Annotated[bool, typer.Argument(help="Whether or not to show class names.")] = False,
):
    """Analyze a given video stream."""
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
    model_sku = YOLO(models[_Model.sku]["path"])
    model_ossa = YOLO(models[_Model.ossa]["path"])

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
    points = ((780, 160), (1840, 490), (1580, 930), (800, 630))
    cap = VideoCapture(source)
    model_sku = YOLO(models[_Model.sku]["path"])
    model_ossa = YOLO(models[_Model.ossa]["path"])

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
