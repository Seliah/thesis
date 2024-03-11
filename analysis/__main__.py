"""Main for analysis module.

This can be used for trying out functionality or hardware, programming or debugging.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from os import getpid
from typing import Optional

import cv2
import typer
from rich.console import Console
from typing_extensions import Annotated

from analysis import state
from analysis.app_logging import logger
from analysis.camera_info import get_sources
from analysis.definitions import GRID_SIZE
from analysis.read import load_motions
from analysis.util.image import draw_grid, show
from analysis.util.tasks import create_task, typer_async
from analysis.vision.capture import analyze_sources
from analysis.vision.motion_search import cli as motion_cli
from analysis.vision.shelf_monitoring import cli as shelf_cli
from user_secrets import URL

console = Console()

app = typer.Typer()
app.add_typer(motion_cli.app, name="motion")
app.add_typer(shelf_cli.yolo_app, name="yolo")
app.add_typer(shelf_cli.shelf_app, name="shelf")

logger.info(f"PID: {getpid()}")


async def _exit_on_input():
    from analysis.util.input import prompt

    logger.info("Running")
    await prompt()
    logger.info("Got input, exiting...")
    state.terminating.set()


@app.command()
@typer_async
async def analyze_all(
    display: Annotated[Optional[str], typer.Argument(help="ID of a camera that is to be visualized.")] = None,
):
    """Analyze the streams of all cameras in the local video system by Adeck Systems."""
    state.motions = load_motions()
    sources = await get_sources()
    task = create_task(analyze_sources(sources, display), "Analysis main task", logger, print_exceptions=True)
    await _exit_on_input()
    await task


@app.command()
@typer_async
async def analyze(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
):
    """Analyze the given stream, saving it with the source as an id."""
    state.motions = load_motions()
    if source is None:
        source = URL
    task = create_task(analyze_sources({source: source}, source), "Analysis main task", logger, print_exceptions=True)
    await _exit_on_input()
    await task


@app.command()
def view(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
    grid: Annotated[bool, typer.Option(help="Show the segment grid.")] = False,
):
    """Show video stream of given source."""
    if source == "0":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        if source is None:
            source = URL
        cap = cv2.VideoCapture(source)
    while True:
        success, image = cap.read()
        if not success:
            console.print("The stream is not returning any valid data.")
            break
        if grid:
            image = draw_grid(image, GRID_SIZE)
        show(image, cap)


if __name__ == "__main__":
    app()
