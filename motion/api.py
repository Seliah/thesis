from __future__ import annotations

from contextlib import asynccontextmanager
from logging import DEBUG, basicConfig, getLogger

from fastapi import FastAPI

import state
from motion.camera_info import get_cameras, get_rtsp_url
from motion.capture import analyze_sources
from motion.read import get_motions_in_area
from util.tasks import create_task

frame_width = 1280
frame_height = 720
cell_width = frame_width / state.GRID_SIZE[0]
cell_height = frame_height / state.GRID_SIZE[1]


basicConfig(level=DEBUG)
_logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _logger.info("Starting analysis.")
    cameras = (await get_cameras())[5:10]
    sources = {camera.uuid: get_rtsp_url(camera) for camera in cameras}
    task = create_task(analyze_sources(sources), "Capture main task", _logger)
    yield
    _logger.info("Terminating analysis.")
    state.terminating = True
    await task


app = FastAPI(lifespan=lifespan)


@app.get("/motion_data")
def get_motions(camera_id: str, x: int, y: int, width: int, height: int):
    cell_y = int(y / cell_height)
    area_height = int(height / cell_height) + 1
    cell_x = int(x / cell_width)
    area_width = int(width / cell_width) + 1
    merged = get_motions_in_area(
        state.motions,
        camera_id,
        cell_x,
        cell_y,
        area_width,
        area_height,
    )
    return merged.nonzero()[1].tolist()
