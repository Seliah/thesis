from pathlib import Path
from typing import List

import typer
from cv2 import imshow, resize, waitKey
from rich.console import Console
from typing_extensions import Annotated
from ultralytics import YOLO
from ultralytics.engine.results import Results

console = Console()
app = typer.Typer()
MODEL_PATH = "weights/yolov8m.pt"
# MODEL_PATH = "weights/yolov8_ossa.pt"
# MODEL_PATH = "weights/yolov8_sku110k.pt"


@app.command()
def info():
    """Print out heatmap data for a given camera."""
    model = YOLO(MODEL_PATH)
    console.print("Classes:")
    console.print(model.names)


@app.command()
def image(
    path: Annotated[Path, typer.Argument(help="The path of the to be analyzed image.")],
    labels: Annotated[bool, typer.Argument(help="Whether or not to show class names.")] = False,
):
    """Analyze a given image."""
    model = YOLO(MODEL_PATH)
    results: List[Results] = model.predict(path)
    for result in results:
        imshow("Results", result.plot(labels=labels, line_width=2))
        waitKey(0)


@app.command()
def stream(
    source: Annotated[str, typer.Argument(help="Video source URL for input stream.")],
    labels: Annotated[bool, typer.Argument(help="Whether or not to show class names.")] = False,
):
    """Analyze a given video stream."""
    model = YOLO(MODEL_PATH)
    # for result in model.predict(source, stream=True):
    for result in model.predict(source, stream=True, classes=[0]):
        plotted = result.plot(labels=labels, line_width=2)
        imshow("Results", resize(plotted, (1440, 810)))
        waitKey(1)


if __name__ == "__main__":
    app()
