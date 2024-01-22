"""Module that implements logic to read and interpret motion data from the local state."""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import reduce
from logging import DEBUG, basicConfig
from operator import add
from time import perf_counter
from typing import TYPE_CHECKING, cast

from rich.console import Console

from analysis import definitions, state
from analysis.util.time import today

if TYPE_CHECKING:
    from cv2.typing import Rect
    from scipy.sparse import lil_array

console = Console()


def print_motion_frames(camera_motions: lil_array):
    """Print non zero entries from a given motion sparse matrix."""
    cell_amount = camera_motions.shape[0]
    cells = [camera_motions.getrow(cell_index) for cell_index in range(cell_amount)]
    merged = reduce(add, cells)
    for index in merged.nonzero()[1]:
        console.print(today() + timedelta(seconds=int(index)))


def get_motions_in_area(
    motions: definitions.MotionData,
    camera_id: str,
    bounds_rect: Rect,
) -> lil_array:
    """Get all motion entries in the given cell section."""
    [cell_x, cell_y, cell_width, cell_height] = bounds_rect
    day_id = str(datetime.now(definitions.TIMEZONE).date())
    camera_motions: lil_array = motions[day_id][camera_id]
    indices_rows = [
        [
            *range(
                (cell_y + y) * definitions.GRID_SIZE[1] + cell_x,
                (cell_y + y) * definitions.GRID_SIZE[1] + cell_x + cell_width,
            ),
        ]
        for y in range(cell_height)
    ]
    indices = reduce(add, indices_rows)
    rows = [camera_motions.getrow(index) for index in indices]
    return reduce(add, rows)


def get_motion_data(camera_id: str):
    """Get all recorded motion entries for the given camera."""
    day_id = str(datetime.now(definitions.TIMEZONE).date())
    cams = state.motions.get(day_id, None)
    if cams is None:
        console.print(f"No entries for day {day_id}")
        return None
    camera_motion_data = cams.get(camera_id, None)
    if camera_motion_data is None:
        console.print(f'No entries for source "{camera_id}"')
        return None
    return camera_motion_data


def calculate_heatmap(camera_id: str):
    """Get the count of motions for every segment."""
    if (motion_data := get_motion_data(camera_id)) is None:
        return None
    cell_amount = motion_data.shape[0]
    cells = [motion_data.getrow(cell_index) for cell_index in range(cell_amount)]
    return [cast(int, cell.nnz) for cell in cells]


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    start_time = perf_counter()

    # Get recorded motion data
    day_id = str(datetime.now(definitions.TIMEZONE).date())
    cams = state.motions.get(day_id, None)
    if cams is None:
        console.print(f"No entries for day {day_id}")
    nonz = {camera_id: cam for camera_id, cam in cams.items() if len(cam.nonzero()[1]) != 0}

    # Print recorded motion data
    for camera_id in nonz:
        console.print(f"{camera_id}:")
        camera_motion_data = nonz.get(camera_id, None)
        if camera_motion_data is None:
            console.print("No entries")
        else:
            print_motion_frames(camera_motion_data)
        console.print()

    # Print execution time
    end_time = perf_counter()
    execution_time = end_time - start_time
    console.print(f"The execution time is: {execution_time}s")
