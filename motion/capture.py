from __future__ import annotations

from asyncio import gather, get_event_loop
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging import getLogger

import cv2
import numpy
from cv2.typing import MatLike
from numpy import concatenate, save
from scipy.sparse import lil_array

import state
from motion.read import load_motions
from util.image import draw_grid, draw_overlay
from util.tasks import create_task
from util.time import seconds_since_midnight

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
TIMEFRAMES = int(DAY_IN_SECONDS / state.INTERVAL)

loop = get_event_loop()
executor = ThreadPoolExecutor(None, "Capture")

_logger = getLogger(__name__)


def update_diff_matrix(diff: MatLike, grid_size: tuple[int, int], camera_id: str):
    # Get the dimensions of the image
    height, width = diff.shape

    # Calculate the size of each cell in the grid
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Initialize the boolean matrix
    boolean_matrix = numpy.zeros(grid_size, dtype=bool)

    # Iterate over the cells in the grid
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            # Extract the current cell from the image
            cell = diff[
                y * cell_height : (y + 1) * cell_height,
                x * cell_width : (x + 1) * cell_width,
            ]

            # Check if the cell contains any non-zero values
            has_values = numpy.any(cell != 0)

            if has_values:
                # Update the global matrix
                index_cell = y * grid_size[0] + x
                index_time = int(seconds_since_midnight(datetime.now()) / state.INTERVAL)

                id_day = str(datetime.now().date())
                if id_day not in state.motions:
                    state.motions[id_day] = {}
                day = state.motions[id_day]

                id_cam = camera_id
                if id_cam not in day:
                    day[id_cam] = lil_array((state.CELLS, TIMEFRAMES), dtype=bool)
                camera_motions = day[id_cam]

                camera_motions[index_cell, index_time] = has_values
                # Update the boolean matrix
                boolean_matrix[y, x] = has_values
    return boolean_matrix


def show_four(x1: MatLike, x2: MatLike, x3: MatLike, x4: MatLike):
    top_row = concatenate((x1, x2), axis=1)
    bottom_row = concatenate((x3, x4), axis=1)
    return concatenate((top_row, bottom_row), axis=0)


def analyze_diff(frame: cv2.UMat, reference_frame: cv2.UMat | None, camera_id: str, show: bool):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve motion detection
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Store the first frame as the reference frame
    if reference_frame is None:
        return gray_blurred

    # Compute the absolute difference between the current frame and the reference frame
    frame_diff = cv2.absdiff(reference_frame, gray_blurred)
    # Apply a threshold to identify regions with significant differences
    _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    reference_frame = gray_blurred
    change_matrix = update_diff_matrix(threshold_diff.get(), state.GRID_SIZE, camera_id)
    if show:
        grid = draw_grid(frame.get().copy(), state.GRID_SIZE)
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
    # Return the current frame as the reference frame
    return gray_blurred


def capture_motion(capture: cv2.VideoCapture, camera_id: str, show: bool):
    reference_frame = None
    while capture.isOpened() and not state.terminating:
        success, frame = capture.read()
        if not success:
            break
        # Get UMat from Matlike to use GPU in following calulcations via OpenCL
        frame_umat = cv2.UMat(frame)
        reference_frame = analyze_diff(frame_umat, reference_frame, camera_id, show)
    capture.release()
    cv2.destroyAllWindows()


async def capture(source: str | int, camera_id: str, show: bool):
    capture = cv2.VideoCapture(source)
    await loop.run_in_executor(executor, capture_motion, capture, camera_id, show)


async def capture_sources(sources: dict[str, str], display: str | None = None):
    """Run analysis for all given sources. Save results after termination.

    :param sources: Dictionary that lists the sources as values.
    The keys should be unique identifiers for thes URLs as the analysis results will be saved
    with these keys as IDs.
    :param display: ID for a specific source. When given, the corresponding analysis will be visualized.
    """
    state.motions = load_motions()
    tasks = [
        create_task(
            capture(source, source_id, source_id == display),
            source_id,
            _logger,
        )
        for source_id, source in sources.items()
    ]
    _logger.info("Analysis is running.")
    await gather(*tasks)
    _logger.info("All analysis tasks terminated.")
    with state.PATH_MOTIONS.open("wb") as f:
        save(f, state.motions)
        _logger.info("Wrote analysis results to disk.")
