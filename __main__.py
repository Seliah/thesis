"""Main for analysis module.

This can be used for trying out functionality or hardware, programming or debugging.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from logging import DEBUG, basicConfig, getLogger
from typing import Optional

import cv2
import typer
from typing_extensions import Annotated

import state
from definitions import GRID_SIZE
from motion.camera_info import get_sources
from motion.capture import analyze_sources
from motion.read import calculate_heatmap, get_motion_data, print_motion_frames
from user_secrets import URL
from util.image import draw_grid
from util.input import prompt
from util.tasks import create_task, typer_async

basicConfig(level=DEBUG)
_logger = getLogger()

app = typer.Typer()
read_app = typer.Typer(help="Read saved analysis data.")
app.add_typer(read_app, name="read")


async def _exit_on_input():
    _logger.info("Running")
    await prompt()
    _logger.info("Got input, exiting...")
    state.terminating = True
    state.termination_event.set()


@app.command()
@typer_async
async def analyze_all(
    display: Annotated[Optional[str], typer.Argument(help="ID of a camera that is to be visualized.")] = None,
):
    """Analyze the streams of all cameras in the local video system by Adeck Systems."""
    sources = await get_sources()
    task = create_task(analyze_sources(sources, display), "Capture main task", _logger)
    await _exit_on_input()
    await task


@app.command()
@typer_async
async def analyze(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
):
    """Analyze the given stream, saving it with the source as an id."""
    if source is None:
        source = URL
    task = create_task(analyze_sources({source: source}, source), "Capture main task", _logger)
    await _exit_on_input()
    await task


@app.command()
def view(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
    grid: Annotated[bool, typer.Option(help="Show the segment grid.")] = False,
):
    """Show video stream of given source."""
    if source is None:
        source = URL
    cap = cv2.VideoCapture(0) if source == "0" else cv2.VideoCapture(source)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        _, image = cap.read()
        if grid:
            print(image.shape)
            image = draw_grid(image, GRID_SIZE)
        cv2.imshow("Video", image)
        if cv2.waitKey(1) == ord("q"):
            break


@read_app.command()
def motions(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    """Print out timestamps with motion for the given camera."""
    if source is None:
        source = URL
    if (motion_data := get_motion_data(source)) is None:
        return
    print_motion_frames(motion_data)


@read_app.command()
def heatmap(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    """Print out heatmap data for a given camera."""
    if source is None:
        source = URL
    print(calculate_heatmap(source))


if __name__ == "__main__":
    app()
