"""Module for definition of known datasets and download functionality."""
# Ignore unknown members as roboflow constantly uses those
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from enum import Enum
from typing import TypedDict

import rich
import typer

from user_secrets import API_KEY

console = rich.console.Console()

FORMAT = "yolov8"


class DatasetID(str, Enum):
    """Names of known datasets."""

    sku_gaps = "sku_gaps"


class _DatasetInfo(TypedDict):
    workspace: str
    project: str
    version: int


datasets: dict[DatasetID, _DatasetInfo] = {
    # Dataset that annotated empty spaces in shelves
    # Source: Final Project Object detection for instore inventory management, “Empty spaces in a supermarket hanger Dataset,” Roboflow Universe. Roboflow, Aug. 2023. Available: https://universe.roboflow.com/final-project-object-detection-for-instore-inventory-management/empty-spaces-in-a-supermarket-hanger-1upsp. [Accessed: Apr. 09, 2024]
    # Based on SKU110k, based in the following paper: E. Goldman et al., “Precise Detection in Densely Packed Scenes.” arXiv, Apr. 30, 2019. Available: http://arxiv.org/abs/1904.00853. [Accessed: Jan. 17, 2024]
    # Hosted on Roboflow: B. Dwyer, J. Nelson, T. Hansen, and et. al., “Roboflow (Version 1.0) [Software].” 2024. Available: https://roboflow.com. [Accessed: Apr. 09, 2024]
    DatasetID.sku_gaps: {
        "workspace": "final-project-object-detection-for-instore-inventory-management",
        "project": "empty-spaces-in-a-supermarket-hanger-1upsp",
        # Older versions seem to contain more images but falsy annotations in some of them
        "version": 29,
    },
}
"""Defintion for all known datasets.

This dict can be appended with new datasets to make them downloadable.
"""


def download(dataset_id: DatasetID):
    """Script to download a given known dataset."""
    from roboflow import Roboflow

    dataset = datasets[dataset_id]
    workspace = Roboflow(api_key=API_KEY).workspace(dataset["workspace"])
    project = workspace.project(dataset["project"])
    project.version(dataset["version"]).download(FORMAT, f"datasets/{dataset['project']}")
    console.print(
        "Now you need to go to ... and edit the paths to be accurate!",
        "Else there are gonna be errors when starting the training process.",
    )


if __name__ == "__main__":
    typer.run(download)
