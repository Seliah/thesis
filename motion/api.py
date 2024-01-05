from contextlib import asynccontextmanager
from logging import DEBUG, basicConfig, getLogger

from fastapi import FastAPI

import state
from motion.capture import GRID_SIZE, main, motions
from motion.read import get_motions_in_area
from util.tasks import create_task

frame_width = 1280
frame_height = 720
cell_width = frame_width / GRID_SIZE[0]
cell_height = frame_height / GRID_SIZE[1]


basicConfig(level=DEBUG)
_logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Start")
    task = create_task(main(), "Capture main task", _logger)
    yield
    print("End")
    state.terminating = True
    await task


app = FastAPI(lifespan=lifespan)


@app.get("/motion_data")
def get_motions(camera_id: str, x: int, y: int, width: int, height: int) -> list[int]:
    cell_y = int(y / cell_height)
    area_height = int(height / cell_height) + 1
    cell_x = int(x / cell_width)
    area_width = int(width / cell_width) + 1
    merged = get_motions_in_area(
        motions, camera_id, cell_x, cell_y, area_width, area_height
    )
    return merged.nonzero()[1].tolist()
