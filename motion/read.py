from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from logging import DEBUG, basicConfig, getLogger
from operator import add
from time import perf_counter
from typing import Any

from numpy import load
from scipy.sparse import lil_array

import state
from util.time import today

_logger = getLogger(__name__)


def load_motions():
    try:
        _logger.debug(f"Loading motion data from {state.PATH_MOTIONS}.")
        with state.PATH_MOTIONS.open("rb") as f:
            return load(f, allow_pickle=True).item()
    except FileNotFoundError:
        _logger.info("No saved motion data was found, starting fresh.")
        return {}


def print_motion_frames(camera_motions: lil_array):
    cell_amount = camera_motions.shape[0]
    cells = [camera_motions.getrow(cell_index) for cell_index in range(cell_amount)]
    merged = reduce(add, cells)
    for index in merged.nonzero()[1]:
        print(today() + timedelta(seconds=int(index)))


def get_motions_in_area(
    motions: Any,
    camera_id: str,
    cell_x: int,
    cell_y: int,
    cell_width: int,
    cell_height: int,
) -> lil_array:
    day_id = str(datetime.now().date())
    camera_motions: lil_array = motions[day_id][camera_id]
    indices_rows = [
        [
            *range(
                (cell_y + y) * state.GRID_SIZE[1] + cell_x,
                (cell_y + y) * state.GRID_SIZE[1] + cell_x + cell_width,
            ),
        ]
        for y in range(cell_height)
    ]
    indices = reduce(add, indices_rows)
    rows = [camera_motions.getrow(index) for index in indices]
    # rows = [camera_motions.getrow(index) for index in indices]
    return reduce(add, rows)


def calculate_heatmap(
    motions: Any,
    camera_id: str,
) -> list[int]:
    day_id = str(datetime.now().date())
    camera_motions: lil_array = motions[day_id][camera_id]
    cell_amount = camera_motions.shape[0]
    cells = [camera_motions.getrow(cell_index) for cell_index in range(cell_amount)]
    return [cell.nnz for cell in cells]


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    # motions = get_motions_in_area("cam", 5, 3, 1, 1)
    # for index in motions.nonzero()[1]:
    #     print(today() + timedelta(seconds=int(index)))
    motions = load_motions()
    start_time = perf_counter()
    day_id = str(datetime.now().date())
    cams: dict[str, lil_array] = motions[day_id]
    nonz = {camera_id: cam for camera_id, cam in cams.items() if len(cam.nonzero()[1]) != 0}
    for camera_id in nonz.keys():
        print(f"{camera_id}:")
        print_motion_frames(motions[day_id][camera_id])
        print()
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}s")
