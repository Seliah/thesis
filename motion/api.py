from fastapi import FastAPI

from motion.capture import GRID_SIZE
from motion.read import get_motions_in_area

app = FastAPI()

frame_width = 1280
frame_height = 720
cell_width = frame_width / GRID_SIZE[0]
cell_height = frame_height / GRID_SIZE[1]


@app.get("/motions")
def get_motions(camera_id: str, x: int, y: int, width: int, height: int):
    cell_y = int(y / cell_height)
    area_height = int(height / GRID_SIZE[0]) + 1
    cell_x = int(x / cell_width)
    area_width = int(width / GRID_SIZE[1]) + 1
    motions = get_motions_in_area(camera_id, cell_x, cell_y, area_width, area_height)
    return cell_width
