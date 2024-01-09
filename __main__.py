from logging import DEBUG, basicConfig, getLogger
from typing import Optional

import typer
from typing_extensions import Annotated

import state
from motion.camera_info import get_cameras, get_rtsp_url
from motion.capture import analyze_sources
from util.input import prompt
from util.tasks import create_task, typer_async

basicConfig(level=DEBUG)
_logger = getLogger()

app = typer.Typer()


async def exit_on_input():
    _logger.info("Running")
    await prompt()
    _logger.info("Got input, exiting...")
    state.terminating = True


async def get_sources():
    return {camera.uuid: get_rtsp_url(camera) for camera in await get_cameras()}


@app.command()
@typer_async
async def all(
    display: Annotated[Optional[str], typer.Argument(help="ID of a camera that is to be visualized.")] = None,
):
    sources = await get_sources()
    task = create_task(analyze_sources(sources, display), "Capture main task", _logger)
    await exit_on_input()
    await task


@app.command()
@typer_async
async def direct(
    source: Annotated[str, typer.Argument(help="Video source. Can be a RTSP URL.")],
):
    task = create_task(analyze_sources({source: source}, source), "Capture main task", _logger)
    await exit_on_input()
    await task


if __name__ == "__main__":
    app()
