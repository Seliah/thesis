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

cap = cv2.VideoCapture(URL)

GRID_SIZE = (9, 16)
"""How many rows and columns should exist to define the cells."""
INTERVAL = 1

DAY_IN_SECONDS = int(timedelta(days=1).total_seconds())
timeframes = int(DAY_IN_SECONDS / INTERVAL)

# a = zeros((9, 16, timeframes), dtype=bool)
a = [
    [lil_array((9 * 16, timeframes), dtype=bool) for _j in range(70)] for _d in range(5)
]

reference_frame = None


def generate_boolean_matrix(image: MatLike, grid_size: tuple[int, int] = (9, 16)):
    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the size of each cell in the grid
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Initialize the boolean matrix
    boolean_matrix = numpy.zeros(grid_size, dtype=bool)

    # Iterate over the cells in the grid
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            # Extract the current cell from the image
            cell = image[
                y * cell_height : (y + 1) * cell_height,
                x * cell_width : (x + 1) * cell_width,
            ]

            # Check if the cell contains any non-zero values
            has_values = numpy.any(cell != 0)

            # Update the boolean matrix
            boolean_matrix[y, x] = has_values
            # Update the global matrix
            index = int(seconds_since_midnight() / INTERVAL)
            a[4][69][y * grid_size[0] + x, index] = has_values
    return boolean_matrix


while cap.isOpened():
    success, frame = cap.read()
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
    # Iterate over the contours and perform further processing as needed

    grid = draw_grid(frame.copy(), GRID_SIZE)
    change_matrix = generate_boolean_matrix(threshold_diff, GRID_SIZE)
    overlayed = draw_overlay(grid, change_matrix)

    # Display the results or perform other actions based on motion detection
    top_row = concatenate(
        (frame, cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)), axis=1
    )
    bottom_row = concatenate(
        (
            cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR),
            overlayed,
        ),
        axis=1,
    )
    merged = concatenate((top_row, bottom_row), axis=0)
    cv2.imshow("Motion Detection", cv2.resize(merged, (1920, 1080)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        with Path("motions.npy").open("wb") as f:
            save(f, a)
        break

cap.release()
cv2.destroyAllWindows()
