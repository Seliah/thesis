from enum import Enum
from pathlib import Path
from typing import Dict, List, TypedDict, cast

import typer
from cv2 import imshow, rectangle, resize, waitKey
from cv2.typing import Rect
from rich.console import Console
from torchvision.ops import box_iou
from typing_extensions import Annotated
from ultralytics import YOLO
from ultralytics.engine.results import Results

console = Console()
app = typer.Typer()


class Model(str, Enum):
    ossa = "ossa"
    sku = "sku110k"
    m = "m"


class ModelInfo(TypedDict):
    info: str
    """A description of the given model."""
    path: Path
    """The path of the weights file."""


models: Dict[str, ModelInfo] = {
    Model.ossa: {
        "info": "qwer",
        "path": Path("weights/yolov8_ossa.pt"),
    },
    Model.sku: {
        "info": "asdf",
        "path": Path("weights/yolov8_sku110k.pt"),
    },
    Model.m: {
        "info": "yxcv",
        "path": Path("weights/yolov8m.pt"),
    },
}


@app.command()
def info(model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")]):
    """Print out heatmap data for a given camera."""
    model = YOLO(models[model_id]["path"])
    console.print("Classes:")
    console.print(models[model_id]["info"])
    console.print(model.names)


@app.command()
def image(
    model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")],
    path: Annotated[Path, typer.Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, typer.Option(help="Whether or not to show class names.")] = False,
):
    """Analyze a given image."""
    model = YOLO(models[model_id]["path"])
    results = cast(List[Results], model.predict(path))
    for result in results:
        # result.
        console.print(result.boxes)
        # console.print(result.numpy())
        imshow("Results", result.plot(labels=labels, line_width=2))
        waitKey(0)


@app.command()
def stream(
    source: Annotated[str, typer.Argument(help="Video source URL for input stream.")],
    model_id: Annotated[Model, typer.Argument(help="Which model to use for analysis.")],
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
    model_sku = YOLO(models[Model.sku]["path"])
    result_sku = cast(List[Results], model_sku.predict(path_img1))[0]
    sku_boxes = result_sku.boxes
    if sku_boxes is None:
        raise Exception

    model_ossa = YOLO(models[Model.ossa]["path"])
    result_ossa = cast(List[Results], model_ossa.predict(path_img2))[0]
    ossa_boxes = result_ossa.boxes
    if ossa_boxes is None:
        raise Exception

    overlapping = box_iou(result_ossa.boxes.xyxy, result_sku.boxes.xyxy)
    for gap_index, item_index in overlapping.nonzero():
        overlap = overlapping[gap_index, item_index]
        if overlap > overlap_threshold:
            box = cast(List[float], result_sku.boxes.xyxy[item_index].tolist())
            rectangle(result_ossa.orig_img, get_rect_from_box(box), (0, 255, 0), 2)

    imshow("Results", result_ossa.orig_img)
    waitKey(0)


def get_rect_from_box(box: List[float]) -> Rect:
    coords = (box[0], box[1], box[2] - box[0], box[3] - box[1])
    return [int(coord) for coord in coords]


if __name__ == "__main__":
    app()
