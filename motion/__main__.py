from datetime import timedelta
from pathlib import Path

import cv2
import numpy
from cv2.typing import MatLike
from numpy import concatenate, save
from scipy.sparse import lil_array

from user_secrets import URL
from util.image import draw_grid, draw_overlay
from util.time import seconds_since_midnight

GRID_SIZE = (9, 16)
"""How many rows and columns should exist to define the cells."""
INTERVAL = 1

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
timeframes = int(DAY_IN_SECONDS / INTERVAL)

a = [
    [lil_array((9 * 16, timeframes), dtype=bool) for _j in range(70)] for _d in range(5)
]

reference_frame = None


def update_diff_matrix(diff: MatLike, grid_size: tuple[int, int] = (9, 16)):
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

            # Update the global matrix
            index = int(seconds_since_midnight() / INTERVAL)
            a[4][69][y * grid_size[0] + x, index] = has_values
            # Update the boolean matrix
            boolean_matrix[y, x] = has_values
    return boolean_matrix


def show_four(x1: MatLike, x2: MatLike, x3: MatLike, x4: MatLike):
    top_row = concatenate((x1, x2), axis=1)
    bottom_row = concatenate((x3, x4), axis=1)
    return concatenate((top_row, bottom_row), axis=0)


def capture_motion(capture: cv2.VideoCapture):
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        grid = draw_grid(frame.copy(), GRID_SIZE)
        change_matrix = update_diff_matrix(threshold_diff, GRID_SIZE)
        overlayed = draw_overlay(grid, change_matrix)

        # Display the results or perform other actions based on motion detection
        merged = show_four(
            frame,
            cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR),
            overlayed,
        )
        cv2.imshow("Motion Detection", cv2.resize(merged, (1920, 1080)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            with Path("motions.npy").open("wb") as f:
                save(f, a)
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture = cv2.VideoCapture(URL)
    merged = capture_motion(capture)
