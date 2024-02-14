"""Module for deep learning training call."""
from multiprocessing import cpu_count
from typing import Annotated

import rich
from typer import Argument, Option, Typer

from scripts.datasets import DatasetID, datasets

app = Typer(help="Train a YOLOv8 model.")


BATCH_SIZE_HELP = ("Batch size to use for training. It's best to keep this at -1 for auto until you get an error"
          "- use a slightly lower value than calculated if you have problems.")

DEFAULT_KWARGS = {
    "patience": 25,
    "batch": 6,
    "workers": cpu_count(),
}


def _prep():
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
    model = _prep()
    dataset = datasets[dataset_id]
    # See https://docs.ultralytics.com/modes/train/
    model.train(data=f"datasets/{dataset['project']}/data.yaml", time=hours, **DEFAULT_KWARGS)  # pyright: ignore[reportUnknownMemberType]

@app.command()
def epochs(
    dataset_id: Annotated[DatasetID, Argument(help="What dataset to train with.")],
    epochs: Annotated[int, Argument(help="How many epochs to train for.")],
    batch_size: Annotated[int, Option(help=BATCH_SIZE_HELP)] = -1,
):
    """Train for the given amount of epochs."""
    model = _prep()
    dataset = datasets[dataset_id]
    # See https://docs.ultralytics.com/modes/train/
    model.train(data=f"datasets/{dataset['project']}/data.yaml", epochs=epochs, **DEFAULT_KWARGS)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    app()
