from datetime import datetime
from logging import DEBUG, basicConfig, getLogger
from typing import Optional

import cv2
import typer
from typing_extensions import Annotated

import state
from motion.camera_info import get_sources
from motion.capture import analyze_sources
from motion.read import calculate_heatmap, load_motions, print_motion_frames
from user_secrets import URL
from util.input import prompt
from util.tasks import create_task, typer_async

basicConfig(level=DEBUG)
_logger = getLogger()

app = typer.Typer()
read_app = typer.Typer(help="Read saved analysis data.")
app.add_typer(read_app, name="read")


async def exit_on_input():
    _logger.info("Running")
    await prompt()
    _logger.info("Got input, exiting...")
    state.terminating = True
    state.termination_event.set()


@app.command(help="Analyze the streams of all cameras in the local video system by Adeck Systems.")
@typer_async
async def analyze_all(
    display: Annotated[Optional[str], typer.Argument(help="ID of a camera that is to be visualized.")] = None,
):
    sources = await get_sources()
    task = create_task(analyze_sources(sources, display), "Capture main task", _logger)
    await exit_on_input()
    await task


@app.command(help="Analyze the given stream, saving it with the source as an id.")
@typer_async
async def analyze(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
):
    if source is None:
        source = URL
    task = create_task(analyze_sources({source: source}, source), "Capture main task", _logger)
    await exit_on_input()
    await task


@app.command(help="Show video stream of given source.")
def view(
    source: Annotated[Optional[str], typer.Argument(help="Video source. Can be a RTSP URL.")] = None,
):
    if source is None:
        source = URL
    cap = cv2.VideoCapture(source)
    while True:
        _, image = cap.read()
        cv2.imshow("Video", image)
        if cv2.waitKey(1) == ord("q"):
            break


@read_app.command(help="Print out timestamps with motion for the given camera.")
def motions(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    if source is None:
        source = URL
    day_id = str(datetime.now().date())
    motions = load_motions()
    print_motion_frames(motions[day_id][source])


@read_app.command(help="Print out heatmap data for a given camera.")
def heatmap(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    if source is None:
        source = URL
    motions = load_motions()
    print(calculate_heatmap(motions, source))


if __name__ == "__main__":
    app()
