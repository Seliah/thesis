"""Module for defining API app.

This module can be used to run this analysis logic with an HTTP API for communication.
It is intended to run this as a systemd service unit. It is optimized for running long term.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Tuple, cast

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from numpy import uint32
from numpy.typing import NDArray

from analysis import definitions, state
from analysis.app_logging import logger
from analysis.camera_info import get_sources
from analysis.read import load_motions
from analysis.util.tasks import create_task
from analysis.vision.capture import analyze_sources
from analysis.vision.motion_search.read import calculate_heatmap, get_motions_in_area

if TYPE_CHECKING:
    from cv2.typing import Rect

SPAN_SIZE_DOC = (
    "Resolution of returned indices. "
    "'1' would mean that the index of every second of the day that had a motion will be returned. "
    "The maximum return integer would be `24*60*60`. "
    "'30' would mean that the information will be lossy compressed "
    "(the indices now stand for 30 second time slots). "
    "The maximum return integer would be `24*60*2`. "
)


@asynccontextmanager
async def _main(_: FastAPI):
    state.motions = load_motions()
    if not definitions.API_ONLY:
        logger.info("Starting analysis.")
        sources = await get_sources()
        task = create_task(analyze_sources(sources), "Analysis API main task", logger, print_exceptions=True)
        # Wait for program termination command
        yield
        logger.info("Terminating analysis.")
        state.terminating.set()
        await task
    else:
        yield


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
def get_heatmap(camera_id: Annotated[str, Query(description="Identifier of the camera/source in question.")]):
    """Return a list with number for motion occurrences in every segment."""
    return calculate_heatmap(camera_id)


@app.get("/motion_data")
def get_motions_from_percent(  # noqa: PLR0913 we need more params that for API
    camera_id: Annotated[str, Query(description="Identifier of the camera/source in question.")],
    left: Annotated[float, Query(description="Left bound for selection rectangle in percent of total width.")],
    top: Annotated[float, Query(description="Top bound for selection rectangle in percent of total height.")],
    width: Annotated[float, Query(description="Width for selection rectangle in percent of total width")],
    height: Annotated[float, Query(description="Height for selection rectangle in percent of total height.")],
    span_size: Annotated[int, Query(description=SPAN_SIZE_DOC)],
):
    """Get motion frames for given image section (in percent)."""
    y = int(top * definitions.GRID_SIZE[0])
    height = int(height * definitions.GRID_SIZE[0]) + 1
    x = int(left * definitions.GRID_SIZE[1])
    width = int(width * definitions.GRID_SIZE[1]) + 1
    logger.debug(f"{x}, {y} - {width}, {height}")
    return get_motions_from_cells(camera_id, (x, y, width, height), span_size)


def get_motions_from_cells(camera_id: str, bounds: Rect, span_size: int = 1):
    """Get motion frames for given image section (in cells)."""
    (x, y, width, height) = bounds
    if (x + width) > definitions.GRID_SIZE[1] or (y + height) > definitions.GRID_SIZE[0]:
        logger.error(f"{x}, {y} - {width}, {height}")
        raise HTTPException(422, "Requested selection is out of bounds.")
    merged = get_motions_in_area(state.motions, camera_id, (x, y, width, height))
    motions = cast(Tuple[NDArray[uint32], NDArray[uint32]], merged.nonzero())[1].tolist()
    return {int(index / span_size) for index in motions}
