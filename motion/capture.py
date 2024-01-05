from asyncio import gather, get_event_loop
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path

import cv2
import numpy
from cv2.typing import MatLike
from numpy import concatenate, load, save
from scipy.sparse import lil_array

import state
from user_secrets import URL
from util.image import draw_grid, draw_overlay
from util.tasks import create_task
from util.time import seconds_since_midnight

GRID_SIZE = (9, 16)
"""How many rows and columns should exist to define the cells."""
CELLS = GRID_SIZE[0] * GRID_SIZE[1]
INTERVAL = 1

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
timeframes = int(DAY_IN_SECONDS / INTERVAL)

motions: dict[str, dict[str, lil_array]]

loop = get_event_loop()
executor = ThreadPoolExecutor(None, "Capture")

_logger = getLogger(__name__)

try:
    with Path("motions.npy").open("rb") as f:
        motions = load(f, allow_pickle=True).item()
except FileNotFoundError:
    _logger.info("No saved motion data was found, starting fresh.")
    motions = {}


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
                index_time = int(seconds_since_midnight(datetime.now()) / INTERVAL)

                id_day = str(datetime.now().date())
                if id_day not in motions:
                    motions[id_day] = {}
                day = motions[id_day]

                id_cam = camera_id
                if id_cam not in day:
                    day[id_cam] = lil_array((CELLS, timeframes), dtype=bool)
                camera_motions = day[id_cam]

                camera_motions[index_cell, index_time] = has_values
                # Update the boolean matrix
                boolean_matrix[y, x] = has_values
    return boolean_matrix


def show_four(x1: MatLike, x2: MatLike, x3: MatLike, x4: MatLike):
    top_row = concatenate((x1, x2), axis=1)
    bottom_row = concatenate((x3, x4), axis=1)
    return concatenate((top_row, bottom_row), axis=0)


def capture_motion(capture: cv2.VideoCapture, camera_id: str, show: bool):
    reference_frame = None
    while capture.isOpened() and not state.terminating:
        success, frame = capture.read()
        if not success:
            break

        # Get UMat from Matlike to use GPU in following calulcations via OpenCL
        frame_umat = cv2.UMat(frame)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve motion detection
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # # Store the first frame as the reference frame
        if reference_frame is None:
            reference_frame = gray_blurred
            continue

        # Compute the absolute difference between the current frame and the reference frame
        frame_diff = cv2.absdiff(reference_frame, gray_blurred)
        # Apply a threshold to identify regions with significant differences
        _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Store the current frame as the reference frame
        reference_frame = gray_blurred

        change_matrix = update_diff_matrix(threshold_diff.get(), GRID_SIZE, camera_id)
        if show:
            grid = draw_grid(frame.copy(), GRID_SIZE)
            overlayed = draw_overlay(grid, change_matrix)

            # Display the results or perform other actions based on motion detection
            merged = show_four(
                frame,
                cv2.cvtColor(gray_blurred.get(), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(frame_diff.get(), cv2.COLOR_GRAY2BGR),
                overlayed,
            )
            cv2.imshow("Motion Detection", cv2.resize(merged, (1920, 1080)))
            cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()


async def capture(source: str | int, camera_id: str, show: bool):
    capture = cv2.VideoCapture(source)
    await loop.run_in_executor(executor, capture_motion, capture, camera_id, show)


async def main():
    # set_termination_handler(_service_terminate)
    tasks = [create_task(capture(URL, "cam", True), "cam", _logger)]
    # tasks = [
    #     create_task(
    #         capture(
    #             get_rtsp_url(camera),
    #             camera.uuid,
    #             camera.uuid == "48614000-8267-11b2-8080-2ca59c7596fc",
    #         ),
    #         camera.name,
    #         _logger,
    #     )
    #     for camera in cameras
    # ]
    _logger.info("Running")
    await gather(*tasks)
    _logger.info("done")
    with Path("motions.npy").open("wb") as f:
        save(f, motions)
        _logger.info("Wrote file!")
