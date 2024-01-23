"""Module for definition of known datasets and download functionality."""
# Ignore unknown members as roboflow constantly uses those
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from enum import Enum
from typing import TypedDict

import rich
import typer
from roboflow import Roboflow

from user_secrets import API_KEY

console = rich.console.Console()

FORMAT = "yolov8"


class _DatasetID(str, Enum):
    """Names of known datasets."""

    ossa = "ossa"
    sku = "sku110k"


class _DatasetInfo(TypedDict):
    workspace: str
    project: str
    version: int


datasets: dict[_DatasetID, _DatasetInfo] = {
    _DatasetID.ossa: {
        "workspace": "fyp-ormnr",
        "project": "on-shelf-stock-availability-ox04t",
        "version": 5,
    },
    _DatasetID.sku: {
        "workspace": "jacobs-workspace",
        "project": "sku-110k",
        "version": 4,
    },
}
"""Defintion for all known datasets.

This dict can be appended with new datasets to make them downloadable.
"""


def download(dataset_id: _DatasetID):
    dataset = datasets[dataset_id]
    workspace = Roboflow(api_key=API_KEY).workspace(dataset["workspace"])
    project = workspace.project(dataset["project"])
    project.version(dataset["version"]).download(FORMAT)
    console.print(
        "Now you need to go to ... and edit the paths to be accurate!",
        "Else there are gonna be errors when starting the training process.",
    )


if __name__ == "__main__":
    typer.run(download)
