"""Module for deep learning training call."""
from multiprocessing import cpu_count
from typing import Annotated

import rich
from typer import Argument, Typer

from scripts.datasets import DatasetID, datasets

app = Typer(help="Train a YOLOv8 model.")


DEFAULT_KWARGS = {
    "patience": 20,
    "batch": -1,
    "workers": cpu_count(),
}


def prep():
    rich.print("Booting up! 🚀")
    rich.print("Loading YOLOv8 lib...")
    from ultralytics import YOLO, checks

    # See used device
    checks()
    # Load a base model for training
    return YOLO("weights/yolov8n.pt")


@app.command()
def hours(
    dataset_id: Annotated[DatasetID, Argument(help="What dataset to train with.")],
    hours: Annotated[float, Argument(help="How many hours to train for.")],
):
    """Train for the given time."""
    model = prep()
    dataset = datasets[dataset_id]
    # See https://docs.ultralytics.com/modes/train/
    model.train(data=f"datasets/{dataset['project']}/data.yaml", time=hours, **DEFAULT_KWARGS)  # pyright: ignore[reportUnknownMemberType]


@app.command()
def epochs(
    dataset_id: Annotated[DatasetID, Argument(help="What dataset to train with.")],
    epochs: Annotated[int, Argument(help="How many epochs to train for.")],
):
    """Train for the given amount of epochs."""
    model = prep()
    dataset = datasets[dataset_id]
    # See https://docs.ultralytics.com/modes/train/
    model.train(data=f"datasets/{dataset['project']}/data.yaml", epochs=epochs, **DEFAULT_KWARGS)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    app()
