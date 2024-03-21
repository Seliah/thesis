"""Module for definition of motion_search debug cli.

See __main__ for application.
"""
# Disable __futures__ import hint as it makes typer unfunctional on python 3.8
# ruff: noqa: FA100
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from analysis import state
from analysis.read import load_motions
from analysis.vision.motion_search.read import calculate_heatmap, get_cameras, get_motion_data, print_motion_frames
from user_secrets import URL

console = Console()
app = typer.Typer(help="Read saved analysis data.")


@app.command()
def read(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    """Print out timestamps with motion for the given camera."""
    state.motions = load_motions()
    if source is None:
        source = URL
    if (motion_data := get_motion_data(source)) is None:
        return
    print_motion_frames(motion_data)


@app.command()
def heatmap(
    source: Annotated[Optional[str], typer.Argument(help="Identifier of the to be read camera.")] = None,
):
    """Print out heatmap data for a given camera."""
    state.motions = load_motions()
    if source is None:
        source = URL
    console.print(calculate_heatmap(source))


@app.command()
def cameras():
    """Print out cameras with motion data."""
    state.motions = load_motions()
    cams = get_cameras()
    if cams is None:
        console.print("No cameras for today.")
        return
    console.print(get_cameras())
