"""Module that implements logic for the motion detection function."""
from __future__ import annotations

from datetime import datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from reactivex.operators import do_action, pairwise, throttle_first
from reactivex.operators import map as map_op
from reactivex.subject import Subject
from scipy.sparse import lil_array

from analysis import definitions, state
from analysis.util.image import draw_grid, draw_overlay
from analysis.util.time import seconds_since_midnight

if TYPE_CHECKING:
    from cv2.typing import MatLike
    from numpy.typing import NDArray

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
TIMEFRAMES = int(DAY_IN_SECONDS / definitions.INTERVAL)

FPS = 5
TIME_PER_FRAME = 1 / FPS

_logger = getLogger(__name__)


def _get_changes(diff: MatLike, grid_size: tuple[int, int]):
    """Get the changes represented by the given difference image in a segment matrix."""
    # Get the dimensions of the image
    height, width = diff.shape

    # Calculate the size of each cell in the grid
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Initialize the boolean matrix
    boolean_matrix = np.zeros(grid_size, dtype=bool)

    # Iterate over the cells in the grid
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            # Extract the current cell from the image
            cell = diff[
                y * cell_height : (y + 1) * cell_height,
                x * cell_width : (x + 1) * cell_width,
            ]
            # Check if the cell contains any non-zero values
            if has_values := np.any(cell != 0):
                # Update the boolean matrix
                boolean_matrix[y, x] = has_values
    return boolean_matrix


def analyze_motion(frames: Subject[MatLike], show: bool):
    return frames.pipe(
        # Apply FPS
        throttle_first(TIME_PER_FRAME),
        # Apply image preparation for analysis
        map_op(prepare),
        # Keep the previous frame for diff
        pairwise(),
        map_op(lambda pair: analyze_diff(pair[1][0], pair[1][1], pair[0][1])),
        # Display
        do_action(lambda t: visualize(t[0], t[1], t[2], t[3]) if show else None),
        map_op(lambda t: t[3]),
    )


def update_global_matrix(
    change_matrix: NDArray[Any],
    camera_id: str,
):
    """Update the global motion store with the given segment matrix."""
    non_zero = change_matrix.nonzero()
    for index, y in enumerate(non_zero[0]):
        x = non_zero[1][index]
        # Update the global matrix
        index_cell = y * definitions.GRID_SIZE[1] + x
        seconds = seconds_since_midnight(datetime.now(definitions.TIMEZONE))
        index_time = int(seconds / definitions.INTERVAL)

        id_day = str(datetime.now(definitions.TIMEZONE).date())
        if id_day not in state.motions:
            state.motions[id_day] = {}
        day = state.motions[id_day]

        id_cam = camera_id
        if id_cam not in day:
            day[id_cam] = lil_array((definitions.CELLS, TIMEFRAMES), dtype=bool)
        camera_motions = day[id_cam]

        camera_motions[index_cell, index_time] = True


def show_two(x1: MatLike, x2: MatLike):
    """Combine two images horizontally."""
    return np.concatenate((x1, x2), axis=1)


def show_four(x1: MatLike, x2: MatLike, x3: MatLike, x4: MatLike):
    """Combine four images in a 2x2 grid."""
    top_row = show_two(x1, x2)
    bottom_row = show_two(x3, x4)
    return np.concatenate((top_row, bottom_row), axis=0)


def prepare(frame: MatLike):
    """Prepare the given image for GPU based analysis."""
    # Get UMat from Matlike to use GPU in following calulcations via OpenCL
    frame_umat = cv2.UMat(frame)  # type: ignore - this works anyways, the type definition is falsy

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve motion detection
    return frame_umat, cv2.GaussianBlur(gray, (21, 21), 0)


def analyze_diff(original: cv2.UMat, frame: cv2.UMat, reference_frame: cv2.UMat):
    """Get the difference of the given frame in a segment matrix."""
    # Compute the absolute difference between the current frame and the reference frame
    frame_diff = cv2.absdiff(reference_frame, frame)
    # Apply a threshold to identify regions with significant differences
    _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    change_matrix = _get_changes(threshold_diff.get(), definitions.GRID_SIZE)

    # Return the current frame as the reference frame
    return original, frame, frame_diff, change_matrix


def visualize(frame: cv2.UMat, gray_blurred: cv2.UMat, frame_diff: cv2.UMat, change_matrix: NDArray[Any]):
    """Visualize the motion analysis flow."""
    grid = draw_grid(frame.get().copy(), definitions.GRID_SIZE)
    overlayed = draw_overlay(grid, change_matrix)

    # Display the results or perform other actions based on motion detection
    merged = show_four(
        frame.get(),
        cv2.cvtColor(gray_blurred.get(), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(frame_diff.get(), cv2.COLOR_GRAY2BGR),
        overlayed,
    )
    cv2.imshow("Motion Detection", cv2.resize(merged, (1600, 900)))
    cv2.waitKey(1)


def write_motion():
    """Write motion data from local state to disk."""
    with definitions.PATH_MOTIONS.open("wb") as f:
        np.save(f, np.asanyarray(state.motions))
        _logger.info("Wrote motion analysis results to disk.")
