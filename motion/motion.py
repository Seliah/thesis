from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import cv2
import numpy
from cv2.typing import MatLike
from numpy import concatenate
from numpy.typing import NDArray
from scipy.sparse import lil_array

import definitions
from util.image import draw_grid, draw_overlay
from util.time import seconds_since_midnight

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
TIMEFRAMES = int(DAY_IN_SECONDS / definitions.INTERVAL)


def get_changes(diff: MatLike, grid_size: tuple[int, int]):
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
            if has_values := numpy.any(cell != 0):
                # Update the boolean matrix
                boolean_matrix[y, x] = has_values
    return boolean_matrix


def update_global_matrix(
    motions: definitions.MotionData,
    change_matrix: NDArray[Any],
    grid_size: tuple[int, int],
    camera_id: str,
):
    non_zero = change_matrix.nonzero()
    for index, y in enumerate(non_zero[0]):
        x = non_zero[1][index]
        # Update the global matrix
        index_cell = y * grid_size[1] + x
        index_time = int(seconds_since_midnight(datetime.now()) / definitions.INTERVAL)

        id_day = str(datetime.now().date())
        if id_day not in motions:
            motions[id_day] = {}
        day = motions[id_day]

        id_cam = camera_id
        if id_cam not in day:
            day[id_cam] = lil_array((definitions.CELLS, TIMEFRAMES), dtype=bool)
        camera_motions = day[id_cam]

        camera_motions[index_cell, index_time] = True


def show_four(x1: MatLike, x2: MatLike, x3: MatLike, x4: MatLike):
    top_row = concatenate((x1, x2), axis=1)
    bottom_row = concatenate((x3, x4), axis=1)
    return concatenate((top_row, bottom_row), axis=0)


def prepare(frame: MatLike):
    # Get UMat from Matlike to use GPU in following calulcations via OpenCL
    frame_umat = cv2.UMat(frame)  # type: ignore - this works anyways, the type definition is falsy

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve motion detection
    return frame_umat, cv2.GaussianBlur(gray, (21, 21), 0)


def analyze_diff(original: cv2.UMat, frame: cv2.UMat, reference_frame: cv2.UMat):
    # Compute the absolute difference between the current frame and the reference frame
    frame_diff = cv2.absdiff(reference_frame, frame)
    # Apply a threshold to identify regions with significant differences
    _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    change_matrix = get_changes(threshold_diff.get(), definitions.GRID_SIZE)

    # Return the current frame as the reference frame
    return original, frame, frame_diff, change_matrix


def display(frame: cv2.UMat, gray_blurred: cv2.UMat, frame_diff: cv2.UMat, change_matrix: NDArray[Any]):
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
