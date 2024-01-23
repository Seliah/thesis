"""Module for deep learning training call."""
from multiprocessing import cpu_count

import rich
import typer

from scripts.datasets import DatasetID, datasets


def train(dataset_id: DatasetID):
    """Script to train a YOLOv8 deep learning model with the given dataset."""
    rich.print("Booting up! 🚀")
    rich.print("Loading YOLOv8 lib...")
    from ultralytics import YOLO, checks

    # See used device
    checks()

    # Load a base model for training
    model = YOLO("weights/yolov8n.pt")

    # Do the actual training
    dataset = datasets[dataset_id]
    model.train(data=f"datasets/{dataset['project']}/data.yaml", epochs=150, imgsz=416, workers=cpu_count())  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    typer.run(train)
