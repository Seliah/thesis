from datetime import datetime, timedelta
from functools import reduce
from operator import add
from pathlib import Path
from time import perf_counter

from numpy import load
from scipy.sparse import lil_array

from motion.capture import GRID_SIZE
from util.time import today


def load_motions():
    with Path("motions.npy").open("rb") as f:
        return load(f, allow_pickle=True).item()


def print_motion_frames(camera_motions: lil_array):
    cell_amount = camera_motions.shape[0]
    cells = [camera_motions.getrow(cell_index) for cell_index in range(cell_amount)]
    merged = reduce(add, cells)
    for index in merged.nonzero()[1]:
        print(today() + timedelta(seconds=int(index)))


def get_motions_in_area(
    camera_id: str, cell_x: int, cell_y: int, cell_width: int, cell_height: int
):
    motions = load_motions()
    day_id = str(datetime.now().date())
    camera_motions: lil_array = motions[day_id][camera_id]
    indices_rows = [
        [
            *range(
                (cell_y + y) * GRID_SIZE[1] + cell_x,
                (cell_y + y) * GRID_SIZE[1] + cell_x + cell_width,
            )
        ]
        for y in range(cell_height)
    ]
    indices = reduce(add, indices_rows)
    rows = [camera_motions.getrow(index) for index in indices]
    merged = reduce(add, rows)
    for index in merged.nonzero()[1]:
        print(today() + timedelta(seconds=int(index)))


if __name__ == "__main__":
    # get_motions_in_area("cam", 5, 3, 1, 1)
    motions = load_motions()
    start_time = perf_counter()
    day_id = str(datetime.now().date())
    cams: dict[str, lil_array] = motions[day_id]
    nonz = {
        camera_id: cam for camera_id, cam in cams.items() if len(cam.nonzero()[1]) != 0
    }

    for camera_id in nonz.keys():
        print(f"{camera_id}:")
        print_motion_frames(motions[day_id][camera_id])
        print()
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}s")
