"""Module for defining API app.

This module can be used to run this analysis logic with an HTTP API for communication.
It is intended to run this as a systemd service unit. It is optimized for running long term.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from logging import DEBUG, basicConfig, getLogger
from typing import List, Tuple, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numpy import uint32
from numpy.typing import NDArray

from analysis import definitions, state
from analysis.camera_info import get_sources
from analysis.read import load_motions
from analysis.util.tasks import create_task
from analysis.vision.capture import analyze_sources
from analysis.vision.motion_search.read import calculate_heatmap, get_motions_in_area

basicConfig(level=DEBUG)
_logger = getLogger(__name__)


@asynccontextmanager
async def _main(_: FastAPI):
    state.motions = load_motions()
    _logger.info("Starting analysis.")
    sources = await get_sources()
    task = create_task(analyze_sources(sources), "Capture main task", _logger)
    # Wait for program termination command
    yield
    _logger.info("Terminating analysis.")
    state.terminating.set()
    await task


app = FastAPI(lifespan=_main)

# Set CORS headers to enable requests from a browser
# See https://fastapi.tiangolo.com/tutorial/cors/
ORIGINS = [
    "https://localhost:4200",
    "https://adeck",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/heatmap")
def get_heatmap(camera_id: str):
    """Return a list with number for motion occurrences in every segment."""
    return calculate_heatmap(camera_id)


@app.get("/motion_data/percent")
def get_motions_from_percent(camera_id: str, left: float, top: float, width: float, height: float):
    """Get motion frames for given image section (in percent)."""
    y = int(top * definitions.GRID_SIZE[0])
    height = int(height * definitions.GRID_SIZE[0]) + 1
    x = int(left * definitions.GRID_SIZE[1])
    width = int(width * definitions.GRID_SIZE[1]) + 1
    _logger.debug(f"{x}, {y} - {width}, {height}")
    return get_motions_from_cells(camera_id, x, y, width, height)


@app.get("/motion_data/pixels")
def get_motions_from_pixels(camera_id: str, x_pixels: int, y_pixels: int, width_pixels: int, height_pixels: int):
    """Get motion frames for given image section (in pixels)."""
    # TODO(elias): use actual resolution
    frame_width = 640
    frame_height = 360
    cell_width = frame_width / definitions.GRID_SIZE[0]
    cell_height = frame_height / definitions.GRID_SIZE[1]
    y = int(y_pixels / cell_height)
    height = int(height_pixels / cell_height) + 1
    x = int(x_pixels / cell_width)
    width = int(width_pixels / cell_width) + 1
    return get_motions_from_cells(camera_id, x, y, width, height)


@app.get("/motion_data/cells")
def get_motions_from_cells(camera_id: str, x: int, y: int, width: int, height: int) -> List[int]:  # noqa: UP006
    """Get motion frames for given image section (in cells)."""
    if (x + width) > definitions.GRID_SIZE[1] or (y + height) > definitions.GRID_SIZE[0]:
        _logger.error(f"{x}, {y} - {width}, {height}")
        raise HTTPException(422, "Requested selection is out of bounds.")
    merged = get_motions_in_area(state.motions, camera_id, (x, y, width, height))
    asdf = cast(Tuple[NDArray[uint32], NDArray[uint32]], merged.nonzero())
    return asdf[1].tolist()
